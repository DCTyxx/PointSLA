"""Trainer for Step-1 CDC base-geometry pretraining.

It reuses the frozen CLIP text encoder and prompt template from GFS-VL, but it
strictly restricts Step 1 to base-class data and base-class names. No novel
few-shot samples, registration sets, or novel class names are allowed here.

The trainer also evaluates three base-only geometry diagnostics on the held-out
base validation split:
- TAS: Tangent Alignment Score.
- GDG: Geometry Discrimination Gap.
- SVR: Semantic Valid Rate.
"""

import torch
import torch.utils.data

from pcseg.config import cfg as pla_cfg
from pcseg.config import cfg_from_list, cfg_from_yaml_file
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models.PLA.pcseg.models import build_text_network
from pointcept.models.PLA.pcseg.models.text_networks import (
    load_text_embedding_from_encoder,
)
import pointcept.utils.comm as comm

from .train import TRAINERS, Trainer


@TRAINERS.register_module("CDCBaseGeometryTrainer")
class CDCBaseGeometryTrainer(Trainer):
    """Train CDC geometry on base data with a frozen CLIP semantic prior."""

    def __init__(self, cfg):
        self._assert_base_only_protocol(cfg)
        super().__init__(cfg)
        self._build_and_inject_text_prior()

    @staticmethod
    def _assert_base_only_protocol(cfg):
        names = list(cfg.data.names)
        if len(names) != int(cfg.data.num_classes):
            raise RuntimeError(
                "Step-1 protocol error: data.names must contain exactly the base "
                "classes used by data.num_classes."
            )

        train_type = str(cfg.data.train.type)
        if "BASETrain" not in train_type:
            raise RuntimeError(
                "Step-1 protocol error: CDC base geometry training must use a "
                f"BASETrain dataset, got '{train_type}'."
            )

        if hasattr(cfg, "regis_train_list") and len(cfg.regis_train_list) > 0:
            raise RuntimeError(
                "Step-1 protocol error: registration/few-shot sets must not be "
                "configured during CDC base geometry training."
            )

        for key in cfg.data.keys():
            if str(key).lower().startswith("regis"):
                raise RuntimeError(
                    "Step-1 protocol error: registration/few-shot data entries "
                    f"are forbidden, found cfg.data.{key}."
                )

        geometry_cfg = getattr(cfg, "cdc_geometry_eval", None)
        if geometry_cfg is not None and bool(geometry_cfg.get("enabled", False)):
            val_type = str(cfg.data.val.type)
            if "BASETrain" not in val_type:
                raise RuntimeError(
                    "Step-1 protocol error: CDC geometry evaluation must also use "
                    f"a BASETrain dataset, got '{val_type}'."
                )

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_and_inject_text_prior(self):
        cfg_file = self.cfg.cdc_text_cfg_file
        self.logger.info(
            "=> Building frozen CLIP text prior for CDC base geometry ..."
        )
        cfg_from_yaml_file(cfg_file, pla_cfg)

        # Critical leakage guard: override any class list carried by the PLA
        # yaml and encode ONLY the benchmark base classes from this Step-1 config.
        base_names = list(self.cfg.data.names)
        pla_cfg.CLASS_NAMES = base_names
        cfg_from_list(["TEXT_ENCODER.EXTRACT_EMBED", True], pla_cfg)

        text_encoder = build_text_network(pla_cfg.TEXT_ENCODER).cuda()
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        with torch.no_grad():
            text_embed = load_text_embedding_from_encoder(
                pla_cfg.TEXT_ENCODER,
                text_encoder,
                self.logger,
            )

        if text_embed.shape[0] != len(base_names):
            raise RuntimeError(
                "Step-1 text leakage/protocol error: encoded text prototype count "
                f"{text_embed.shape[0]} != base class count {len(base_names)}."
            )

        model = self._unwrap_model()
        model.set_text_embeddings(text_embed.cuda().float())
        self.logger.info(
            "=> Injected frozen BASE-ONLY text prototypes: "
            f"{tuple(text_embed.shape)} for {len(base_names)} classes"
        )

        del text_encoder
        torch.cuda.empty_cache()

    def before_epoch(self):
        model = self._unwrap_model()
        previous_frozen = getattr(model, "_backbone_is_frozen", None)
        model.set_training_epoch(self.epoch)
        current_frozen = getattr(model, "_backbone_is_frozen", None)

        if (
            previous_frozen is not None
            and current_frozen is not None
            and previous_frozen != current_frozen
        ):
            if current_frozen:
                self.logger.info("=> PTv3 backbone frozen for CDC warm-up.")
            else:
                self.logger.info(
                    "=> PTv3 backbone unfrozen for joint base training with "
                    "the configured small backbone learning rate."
                )

        if self.epoch == model.text_warmup_epochs:
            self.logger.info(
                "=> CDC text-direction constraint is now enabled "
                f"(epoch {self.epoch + 1})."
            )
        super().before_epoch()

    @staticmethod
    def _evenly_spaced_indices(indices, count):
        """Deterministically subsample a 1-D index tensor."""
        if indices.numel() <= count:
            return indices
        positions = torch.linspace(
            0,
            indices.numel() - 1,
            steps=count,
            device=indices.device,
        ).round().long()
        return indices[positions]

    def _build_geometry_feature_bank(self, model, eval_cfg):
        """Build a small, class-balanced feature bank from base validation data."""
        dataset = build_dataset(self.cfg.data.val)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(eval_cfg.get("num_workers", 4)),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        num_classes = int(self.cfg.data.num_classes)
        max_per_class = int(eval_cfg.get("max_points_per_class", 256))
        max_batches = int(eval_cfg.get("max_batches", 32))
        class_features = [[] for _ in range(num_classes)]
        class_counts = [0 for _ in range(num_classes)]

        for batch_idx, input_dict in enumerate(loader):
            if batch_idx >= max_batches:
                break
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            labels = input_dict["segment"].long()
            valid = labels >= 0
            if valid.any() and int(labels[valid].max().item()) >= num_classes:
                raise RuntimeError(
                    "Step-1 geometry evaluation label leakage detected: "
                    "validation contains a non-base class label."
                )

            feature = model.getFeatures(input_dict).float()
            for class_id in range(num_classes):
                remaining = max_per_class - class_counts[class_id]
                if remaining <= 0:
                    continue
                idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                idx = self._evenly_spaced_indices(idx, remaining)
                class_features[class_id].append(feature[idx].detach().cpu())
                class_counts[class_id] += idx.numel()

            if all(count >= max_per_class for count in class_counts):
                break

        bank_features, bank_labels = [], []
        for class_id, chunks in enumerate(class_features):
            if len(chunks) == 0:
                continue
            feat = torch.cat(chunks, dim=0)[:max_per_class]
            bank_features.append(feat)
            bank_labels.append(
                torch.full((feat.shape[0],), class_id, dtype=torch.long)
            )

        if len(bank_features) == 0:
            raise RuntimeError(
                "CDC geometry evaluation could not collect any valid base features."
            )

        features = torch.cat(bank_features, dim=0).cuda(non_blocking=True).float()
        labels = torch.cat(bank_labels, dim=0).cuda(non_blocking=True)
        return features, labels, class_counts

    def _balanced_anchor_indices(self, labels, max_anchors):
        """Deterministically choose approximately class-balanced anchors."""
        classes = torch.unique(labels)
        if classes.numel() == 0:
            return labels.new_empty((0,), dtype=torch.long)
        per_class = max(1, max_anchors // classes.numel())
        selected = []
        for class_id in classes:
            idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
            selected.append(self._evenly_spaced_indices(idx, per_class))
        selected = torch.cat(selected, dim=0)
        if selected.numel() > max_anchors:
            selected = self._evenly_spaced_indices(selected, max_anchors)
        return selected

    @torch.no_grad()
    def _compute_geometry_metrics(self, model, features, labels, eval_cfg):
        """Compute TAS, GDG and SVR on the base-only held-out feature bank."""
        rank = int(model.cdc_rank)
        num_anchors = int(eval_cfg.get("num_anchors", 256))
        knn = int(eval_cfg.get("knn", 16))
        inter_knn = int(eval_cfg.get("inter_knn", 16))
        svr_directions = int(eval_cfg.get("svr_directions", 4))

        anchors = self._balanced_anchor_indices(labels, num_anchors)
        if anchors.numel() == 0:
            raise RuntimeError("CDC geometry evaluation found no anchor points.")

        tas_sum = 0.0
        tas_count = 0
        intra_proj_energy = 0.0
        intra_total_energy = 0.0
        inter_proj_energy = 0.0
        inter_total_energy = 0.0
        svr_valid = 0.0
        svr_total = 0

        for anchor_idx in anchors.tolist():
            z = features[anchor_idx]
            y = labels[anchor_idx]

            same_idx = torch.nonzero(labels == y, as_tuple=False).flatten()
            same_idx = same_idx[same_idx != anchor_idx]
            diff_idx = torch.nonzero(labels != y, as_tuple=False).flatten()
            if same_idx.numel() < rank or diff_idx.numel() == 0:
                continue

            same_dist = (features[same_idx] - z).square().sum(dim=1)
            k_intra = min(knn, same_idx.numel())
            nearest_same = same_idx[torch.topk(same_dist, k=k_intra, largest=False).indices]
            intra_residual = features[nearest_same] - z
            if intra_residual.shape[0] < rank:
                continue

            # Empirical held-out tangent subspace from local same-class residuals.
            _, _, vh = torch.linalg.svd(intra_residual, full_matrices=False)
            q_emp = vh[:rank].transpose(0, 1)  # [D, r]

            # CDC local subspace. SVD orders directions by factor singular value.
            factor = model.get_cdc_factor(
                z.unsqueeze(0), relative_h=model.cdc_probe_h
            )[0]
            q_cdc, _, _ = torch.linalg.svd(factor, full_matrices=False)
            q_cdc = q_cdc[:, :rank]

            overlap = q_cdc.transpose(0, 1) @ q_emp
            tas_sum += float(overlap.square().sum().item() / rank)
            tas_count += 1

            intra_projection = intra_residual @ q_cdc
            intra_proj_energy += float(intra_projection.square().sum().item())
            intra_total_energy += float(intra_residual.square().sum().item())

            inter_dist = (features[diff_idx] - z).square().sum(dim=1)
            k_inter = min(inter_knn, diff_idx.numel())
            nearest_diff = diff_idx[
                torch.topk(inter_dist, k=k_inter, largest=False).indices
            ]
            inter_residual = features[nearest_diff] - z
            inter_projection = inter_residual @ q_cdc
            inter_proj_energy += float(inter_projection.square().sum().item())
            inter_total_energy += float(inter_residual.square().sum().item())

            # Semantic validity is measured with fixed-length normalized CDC
            # directions, so a collapsed factor cannot trivially obtain high SVR.
            clean_scores = model._text_scores(
                z.unsqueeze(0), detach_projector=True
            )
            clean_gt, clean_margin = model._gt_and_margin(
                clean_scores, y.view(1)
            )
            radius = model.text_probe_radius * z.norm().clamp_min(1e-6)
            num_dir = min(svr_directions, q_cdc.shape[1])
            for direction_idx in range(num_dir):
                direction = q_cdc[:, direction_idx]
                delta = radius * direction
                for virtual in (z + delta, z - delta):
                    virtual_scores = model._text_scores(
                        virtual.unsqueeze(0), detach_projector=True
                    )
                    virtual_gt, virtual_margin = model._gt_and_margin(
                        virtual_scores, y.view(1)
                    )
                    valid = (
                        virtual_gt
                        >= clean_gt - model.text_positive_tolerance
                    ) & (
                        virtual_margin
                        >= clean_margin - model.text_margin_tolerance
                    )
                    svr_valid += float(valid.float().item())
                    svr_total += 1

        if tas_count == 0:
            raise RuntimeError(
                "CDC geometry evaluation has no valid anchors; increase the "
                "feature-bank size or reduce CDC rank/knn constraints."
            )

        tas = tas_sum / tas_count
        intra_evr = intra_proj_energy / max(intra_total_energy, 1e-12)
        inter_evr = inter_proj_energy / max(inter_total_energy, 1e-12)
        gdg = intra_evr - inter_evr
        svr = svr_valid / max(svr_total, 1)
        return tas, gdg, svr, tas_count

    def _should_run_geometry_eval(self):
        eval_cfg = getattr(self.cfg, "cdc_geometry_eval", None)
        if eval_cfg is None or not bool(eval_cfg.get("enabled", False)):
            return False
        interval = max(1, int(eval_cfg.get("interval", 5)))
        epoch_number = self.epoch + 1
        return (
            epoch_number % interval == 0
            or epoch_number == self.max_epoch
        )

    @torch.no_grad()
    def _evaluate_base_geometry(self):
        eval_cfg = self.cfg.cdc_geometry_eval
        model = self._unwrap_model()
        was_training = model.training
        model.eval()
        try:
            features, labels, class_counts = self._build_geometry_feature_bank(
                model, eval_cfg
            )
            tas, gdg, svr, anchor_count = self._compute_geometry_metrics(
                model, features, labels, eval_cfg
            )
        finally:
            model.train(was_training)

        self.logger.info(
            "CDC geometry eval (BASE-ONLY) | "
            f"TAS: {tas:.4f} | GDG: {gdg:.4f} | SVR: {svr:.4f} | "
            f"anchors: {anchor_count} | bank: {features.shape[0]}"
        )
        self.logger.info(
            "CDC geometry eval base-class bank counts: "
            + ", ".join(str(int(v)) for v in class_counts)
        )
        if self.writer is not None:
            step = self.epoch + 1
            self.writer.add_scalar("cdc_geometry/TAS", tas, step)
            self.writer.add_scalar("cdc_geometry/GDG", gdg, step)
            self.writer.add_scalar("cdc_geometry/SVR", svr, step)

    def after_epoch(self):
        super().after_epoch()
        if not self._should_run_geometry_eval():
            return

        # Geometry evaluation is intentionally rank-0-only. Other ranks wait so
        # the next training epoch starts with all workers synchronized.
        comm.synchronize()
        if comm.is_main_process():
            self._evaluate_base_geometry()
        comm.synchronize()
