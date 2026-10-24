"""Stage-2 Phase-3 Novel feature augmentation.

The default mode remains CDC-guided augmentation.  For controlled direction
ablations, the same Phase-2 checkpoint, real loss, virtual radius, number of
virtual samples and virtual-loss weight can be reused while changing only the
source of the augmentation direction:

- ``cdc``: learned local CDC tangent subspace (default / main method);
- ``real_only``: no virtual feature, real Base+Novel fine-tuning only;
- ``isotropic``: random unit direction in the full feature space;
- ``random_subspace``: direction in one fixed random rank-r orthonormal subspace;
- ``local_pca``: direction in the empirical local PCA subspace estimated from
  same-class Novel KNN residuals around each anchor.  The target rank is r, but
  when an anchor has fewer than r same-class neighbors the identifiable local
  rank is reduced adaptively instead of inventing random completion directions.

Only the Novel classifier is trainable in Phase 3. PTv3, CDC, the Base head,
semantic projector and radius head remain frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.utils.structure import Point
from .builder import MODELS
from .stage2_phase2_cdc_adapt import Stage2Phase2CDCAdaptSegmentor


@MODELS.register_module()
class Stage2Phase3CDCAugSegmentor(Stage2Phase2CDCAdaptSegmentor):
    """Frozen representation/geometry + Novel-head refinement."""

    VALID_AUGMENTATION_MODES = (
        "cdc",
        "real_only",
        "isotropic",
        "random_subspace",
        "local_pca",
    )

    def __init__(
        self,
        virtual_samples=4,
        virtual_radius=0.025,
        virtual_loss_weight=0.5,
        augmentation_mode="cdc",
        random_subspace_seed=0,
        local_pca_knn=None,
        local_pca_chunk_size=512,
        **kwargs,
    ):
        # Build exactly the Phase-2 hierarchy first so every Phase-2 checkpoint
        # can still be restored with strict=True.
        super().__init__(**kwargs)

        self.virtual_samples = int(virtual_samples)
        self.virtual_radius = float(virtual_radius)
        self.virtual_loss_weight = float(virtual_loss_weight)
        self.augmentation_mode = str(augmentation_mode).lower()
        self.random_subspace_seed = int(random_subspace_seed)
        self.local_pca_knn = int(
            self.tangent_knn if local_pca_knn is None else local_pca_knn
        )
        self.local_pca_chunk_size = int(local_pca_chunk_size)

        if self.augmentation_mode not in self.VALID_AUGMENTATION_MODES:
            raise ValueError(
                f"Unknown augmentation_mode={self.augmentation_mode}; expected one of "
                f"{self.VALID_AUGMENTATION_MODES}"
            )
        if self.virtual_samples < 1:
            raise ValueError("virtual_samples must be >= 1")
        if self.virtual_radius <= 0:
            raise ValueError("virtual_radius must be > 0")
        if self.virtual_loss_weight < 0:
            raise ValueError("virtual_loss_weight must be >= 0")
        if self.local_pca_knn < self.cdc_rank:
            raise ValueError(
                "local_pca_knn must be >= CDC rank so well-supported anchors can "
                f"reach r={self.cdc_rank}; got {self.local_pca_knn}."
            )
        if self.local_pca_chunk_size <= 0:
            raise ValueError("local_pca_chunk_size must be positive")

        # Plain cached tensor on purpose: it is not a model parameter/buffer, so
        # adding this ablation does not change the Phase-2/Phase-3 state_dict and
        # strict checkpoint loading remains valid.
        self._random_subspace_cache = None

        # Phase 3 freezes the task representation, geometry and Base classifier.
        self.backbone.requires_grad_(False)
        self.seg_head.requires_grad_(False)
        self.cdc_head.requires_grad_(False)
        self.semantic_projector.requires_grad_(False)
        self.safe_radius_head.requires_grad_(False)

        # Only the Novel classifier is refined.
        self.novel_seg_head.requires_grad_(True)

        self.freeze_backbone = True
        self.freeze_seg_head = True
        self._assert_phase3_trainability()

    def _assert_phase3_trainability(self):
        frozen_modules = (
            ("backbone", self.backbone),
            ("seg_head", self.seg_head),
            ("cdc_head", self.cdc_head),
            ("semantic_projector", self.semantic_projector),
            ("safe_radius_head", self.safe_radius_head),
        )
        for name, module in frozen_modules:
            if any(p.requires_grad for p in module.parameters()):
                raise RuntimeError(f"{name} must be frozen during Stage-2 Phase-3")

        if not any(p.requires_grad for p in self.novel_seg_head.parameters()):
            raise RuntimeError("Novel segmentation head must be trainable in Phase 3")
        if not all(p.requires_grad for p in self.novel_seg_head.parameters()):
            raise RuntimeError("Every Novel-head parameter must be trainable in Phase 3")

        bad_trainable = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("novel_seg_head.")
        ]
        if bad_trainable:
            raise RuntimeError(
                "Only novel_seg_head may be trainable in Stage-2 Phase-3; got "
                f"{bad_trainable}"
            )

    def train(self, mode=True):
        # Bypass Phase-2 train(), which intentionally makes CDC trainable.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.seg_head.eval()
        self.cdc_head.eval()
        self.semantic_projector.eval()
        self.safe_radius_head.eval()
        self.novel_seg_head.train(mode)
        self._assert_phase3_trainability()
        return self

    def _full_logits(self, feature):
        base_logits = self.seg_head(feature)
        novel_logits = self.novel_seg_head(feature)
        return torch.cat([base_logits, novel_logits], dim=1)

    @torch.no_grad()
    def _cdc_tangent_basis(self, novel_feature):
        """Return the Phase-2 CDC orthonormal tangent basis for Novel features."""
        n = novel_feature.shape[0]
        relative_h = torch.full(
            (n,),
            self.cdc_probe_h,
            device=novel_feature.device,
            dtype=novel_feature.dtype,
        )
        feature_rms = novel_feature.square().mean(dim=-1).add(1e-8).sqrt()
        factor = self.cdc_head(novel_feature, relative_h * feature_rms)
        return self._orthonormalize_factor(factor)

    @torch.no_grad()
    def _fixed_random_subspace_basis(self, feature):
        """Return one deterministic rank-r random orthonormal basis for the run."""
        cache = self._random_subspace_cache
        if (
            cache is None
            or cache.device != feature.device
            or cache.dtype != feature.dtype
        ):
            # Generate on CPU with a private generator so the subspace is fixed
            # and independent of training-time RNG / DDP execution order.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_subspace_seed)
            raw = torch.randn(
                self.feature_dim,
                self.cdc_rank,
                generator=generator,
                dtype=torch.float32,
            )
            q, _ = torch.linalg.qr(raw, mode="reduced")
            cache = q.to(device=feature.device, dtype=feature.dtype)
            self._random_subspace_cache = cache
        return cache

    @torch.no_grad()
    def _local_pca_tangent_basis(self, novel_feature, novel_labels):
        """Fast empirical local PCA basis from same-class Novel KNN residuals.

        This implementation is mathematically equivalent to the previous Local-PCA
        definition but removes two major sources of wasted compute:

        1. KNN is performed class-by-class, so distances to different classes are
           never computed and then masked out.
        2. For residual R in R^{K x D} (K<=32, D=64), PCA is obtained from the
           small Gram matrix G=R R^T in R^{K x K}.  We use batched ``eigh(G)`` and
           recover right singular vectors via V = R^T U Sigma^{-1}.  This avoids
           batched SVD of every K x D residual matrix.

        For each anchor the identifiable rank remains

            r_i = min(cdc_rank, local_pca_knn, K_i),

        with numerically null directions removed. Missing columns are zero padded;
        no random completion directions are introduced. An anchor with no same-class
        neighbor receives a zero basis and therefore a zero virtual step.
        """
        if novel_labels is None:
            raise ValueError("local_pca augmentation requires Novel labels")
        if novel_feature.shape[0] != novel_labels.shape[0]:
            raise ValueError("Novel feature/label size mismatch")

        bank_idx = self._balanced_unique_indices(
            novel_labels, self.tangent_max_bank_points
        )
        bank = novel_feature[bank_idx]
        bank_labels = novel_labels[bank_idx]

        n = novel_feature.shape[0]
        if bank.shape[0] == 0:
            return novel_feature.new_zeros(
                (n, self.feature_dim, self.cdc_rank)
            )

        bases = []
        eps = torch.finfo(novel_feature.dtype).eps
        for start in range(0, n, self.local_pca_chunk_size):
            end = min(start + self.local_pca_chunk_size, n)
            anchors = novel_feature[start:end]
            labels = novel_labels[start:end]
            anchor_ids = torch.arange(start, end, device=novel_feature.device)

            q_emp = novel_feature.new_zeros(
                (end - start, self.feature_dim, self.cdc_rank)
            )

            # Process each semantic class independently. This changes only the
            # implementation cost: Local-PCA has always used same-class KNN.
            for class_label in torch.unique(labels, sorted=False):
                anchor_rows = torch.nonzero(
                    labels.eq(class_label), as_tuple=False
                ).flatten()
                bank_rows = torch.nonzero(
                    bank_labels.eq(class_label), as_tuple=False
                ).flatten()
                if anchor_rows.numel() == 0 or bank_rows.numel() == 0:
                    continue

                class_anchors = anchors[anchor_rows]
                class_anchor_ids = anchor_ids[anchor_rows]
                class_bank = bank[bank_rows]
                class_bank_ids = bank_idx[bank_rows]

                # Squared Euclidean distance via GEMM. This is substantially
                # cheaper than full-bank cdist followed by a same-class mask.
                anchor_sq = class_anchors.square().sum(dim=1, keepdim=True)
                bank_sq = class_bank.square().sum(dim=1).unsqueeze(0)
                dist = (
                    anchor_sq
                    + bank_sq
                    - 2.0 * torch.matmul(class_anchors, class_bank.transpose(0, 1))
                ).clamp_min_(0.0)

                self_mask = class_anchor_ids[:, None].eq(class_bank_ids[None, :])
                available = (~self_mask).sum(dim=1)
                effective_k = available.clamp(max=self.local_pca_knn).long()
                positive = effective_k > 0
                if not bool(positive.any().item()):
                    continue

                # Only one or two K values normally occur (depending on whether
                # the anchor itself is present in the balanced bank), so this loop
                # stays very small while preserving each anchor's adaptive rank.
                for k_tensor in torch.unique(effective_k[positive], sorted=True):
                    k = int(k_tensor.item())
                    group_mask = effective_k.eq(k)
                    group_rows = anchor_rows[group_mask]
                    group_dist = dist[group_mask].masked_fill(
                        self_mask[group_mask], float("inf")
                    )
                    group_anchor = class_anchors[group_mask]

                    nearest = torch.topk(
                        group_dist,
                        k=k,
                        dim=1,
                        largest=False,
                    ).indices
                    residual = class_bank[nearest] - group_anchor[:, None, :]

                    # Small-Gram PCA: R:[B,K,D] -> G:[B,K,K].  eigh returns
                    # eigenvalues in ascending order, so take/flip the largest.
                    gram = torch.bmm(residual, residual.transpose(1, 2))
                    eigvals, left_vec = torch.linalg.eigh(gram)
                    r_eff = min(self.cdc_rank, k)
                    eigvals = eigvals[:, -r_eff:].flip(dims=(-1,)).clamp_min_(0.0)
                    left_vec = left_vec[:, :, -r_eff:].flip(dims=(-1,))

                    # Recover the corresponding right singular vectors:
                    # V = R^T U / sigma. Numerically null eigen-directions are
                    # explicitly zeroed rather than amplified by division.
                    max_eval = eigvals[:, :1]
                    tol = max_eval * (max(k, self.feature_dim) * eps)
                    valid_dir = eigvals > tol
                    sigma = eigvals.sqrt()
                    denom = sigma.clamp_min(1e-12).unsqueeze(1)
                    right_vec = torch.bmm(
                        residual.transpose(1, 2), left_vec
                    ) / denom
                    right_vec = F.normalize(right_vec, p=2, dim=1, eps=1e-8)
                    right_vec = right_vec * valid_dir[:, None, :].to(right_vec.dtype)
                    right_vec = torch.nan_to_num(right_vec)

                    q_emp[group_rows, :, :r_eff] = right_vec

            bases.append(q_emp)

        return torch.cat(bases, dim=0)

    @torch.no_grad()
    def _sample_subspace_directions(self, q, n):
        """Sample M unit directions from per-anchor or shared orthonormal Q.

        For adaptive Local-PCA, unused basis columns are exactly zero.  Random
        coefficients on those columns vanish after projection, and the final
        normalization therefore samples only inside the available empirical
        subspace.  A completely zero basis yields a zero direction.
        """
        coeff = torch.randn(
            n,
            self.virtual_samples,
            self.cdc_rank,
            device=q.device,
            dtype=q.dtype,
        )
        coeff = F.normalize(coeff, p=2, dim=-1, eps=1e-8)
        if q.ndim == 2:
            direction = torch.einsum("dr,nmr->nmd", q, coeff)
        else:
            direction = torch.einsum("ndr,nmr->nmd", q, coeff)
        return F.normalize(direction, p=2, dim=-1, eps=1e-8)

    @torch.no_grad()
    def _sample_virtual_novel_features(self, novel_feature, novel_labels=None):
        """Generate virtual Novel features under the selected direction ablation."""
        n, d = novel_feature.shape

        if self.augmentation_mode == "cdc":
            q = self._cdc_tangent_basis(novel_feature)
            direction = self._sample_subspace_directions(q, n)
        elif self.augmentation_mode == "isotropic":
            direction = torch.randn(
                n,
                self.virtual_samples,
                d,
                device=novel_feature.device,
                dtype=novel_feature.dtype,
            )
            direction = F.normalize(direction, p=2, dim=-1, eps=1e-8)
        elif self.augmentation_mode == "random_subspace":
            q = self._fixed_random_subspace_basis(novel_feature)
            direction = self._sample_subspace_directions(q, n)
        elif self.augmentation_mode == "local_pca":
            q = self._local_pca_tangent_basis(novel_feature, novel_labels)
            direction = self._sample_subspace_directions(q, n)
        else:
            raise RuntimeError(
                f"Virtual feature sampling called for mode={self.augmentation_mode}"
            )

        feature_norm = novel_feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        step = self.virtual_radius * feature_norm[:, None, :] * direction
        virtual = novel_feature[:, None, :] + step
        return virtual.reshape(-1, d), step

    def forward(self, input_dict):
        # PTv3 is fixed in Phase 3. Keep geometry-side computation in fp32 to
        # match Phase 2 and avoid AMP-dependent direction changes.
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                point = self.backbone(Point(input_dict))
                feature = point.feat if isinstance(point, Point) else point
                feature = feature.detach().float()

        # Standard inference path: augmentation is not needed at test time.
        if not self.training:
            seg_logits = self._full_logits(feature)
            if "segment" in input_dict:
                labels = input_dict["segment"].long().to(seg_logits.device)
                loss = self.criteria(seg_logits, labels)
                return dict(loss=loss, seg_logits=seg_logits)
            return dict(seg_logits=seg_logits)

        labels = input_dict["segment"].long().to(feature.device)
        valid = labels >= 0
        if not bool(valid.any().item()):
            raise RuntimeError("Phase-3 batch contains no trusted GT points")

        # Only trusted Base GT + inserted Novel GT supervise Phase 3.
        real_feature = feature[valid]
        real_labels = labels[valid]
        total_classes = self.num_base_classes + self.num_novel_classes
        if int(real_labels.max().item()) >= total_classes:
            raise RuntimeError(
                "Phase-3 trusted label exceeds Base+Novel classifier range: "
                f"max_label={int(real_labels.max().item())}, "
                f"num_classes={total_classes}"
            )

        real_logits = self._full_logits(real_feature)
        real_loss = self.criteria(real_logits, real_labels)

        novel_mask = real_labels >= self.num_base_classes
        num_novel = novel_mask.sum()
        use_virtual = self.augmentation_mode != "real_only"

        if use_virtual and bool(novel_mask.any().item()):
            novel_feature = real_feature[novel_mask]
            novel_labels = real_labels[novel_mask]

            virtual_feature, step = self._sample_virtual_novel_features(
                novel_feature, novel_labels
            )
            virtual_labels = novel_labels.repeat_interleave(self.virtual_samples)
            virtual_logits = self._full_logits(virtual_feature)
            virtual_loss = self.criteria(virtual_logits, virtual_labels)

            relative_step = (
                step.norm(dim=-1)
                / novel_feature.norm(dim=-1).clamp_min(1e-8)[:, None]
            ).mean()
            num_virtual = virtual_labels.numel()
        else:
            # Real-only ablation or a crop with no Novel point.
            virtual_loss = real_loss.new_zeros(())
            relative_step = real_loss.new_zeros(())
            num_virtual = 0

        loss = real_loss + self.virtual_loss_weight * virtual_loss

        return {
            "loss": loss,
            "real_loss": real_loss.detach(),
            "virtual_loss": virtual_loss.detach(),
            "num_novel_points": num_novel.to(dtype=real_loss.dtype),
            "num_virtual_points": real_loss.new_tensor(float(num_virtual)),
            "virtual_step_ratio": relative_step.detach(),
        }
