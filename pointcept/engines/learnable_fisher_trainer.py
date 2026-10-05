import torch
import torch.nn.functional as F

from .learnable_fisher import LearnableCovarianceFisher
from .train import GFS_VL_Trainer, TRAINERS
import pointcept.utils.comm as comm


@TRAINERS.register_module("GFS_VL_LearnableFisher_Trainer")
class GFS_VL_LearnableFisher_Trainer(GFS_VL_Trainer):
    """GFS-VL trainer with Fisher geometry regularizing backbone features.

    Pseudo-label selection and adaptive infilling keep the original cosine
    calibration path. In parallel, novel-class support points are encoded by
    the trainable point-cloud backbone (under no-grad) to estimate class moments
    and construct a support-conditioned Fisher geometry

        G_c^F = (Sigma_c + eps I)^(-1).

    A lightweight generator predicts

        A_c = h_theta(mu_c, Lambda_c, R_c)

    and deforms the Fisher-normalized geometry as

        G_c^LF = G_c^{F/2}(I + alpha A_c A_c^T)G_c^{F/2}.

    During training, query backbone features are NOT detached. Therefore the
    Fisher metric CE directly regularizes the trainable point-cloud backbone,
    while the support-side statistics remain detached/fixed within each
    registration run.
    """

    def build_zero_shot_model(self):
        # Build the original GFS-VL semantic matcher first. With the current
        # config this is cosine and remains the calibration/infilling metric.
        super().build_zero_shot_model()
        self.learnable_fisher_matcher = None

        if not bool(getattr(self.cfg, "metric_fisher_learnable", True)):
            return

        feat_dim = int(
            getattr(
                self.cfg,
                "metric_fisher_feat_dim",
                getattr(self.cfg.model, "backbone_out_channels", 64),
            )
        )
        matcher = LearnableCovarianceFisher(
            feat_dim=feat_dim,
            spectral_rank=int(
                getattr(self.cfg, "metric_fisher_spectral_rank", 32)
            ),
            low_rank=int(getattr(self.cfg, "metric_fisher_low_rank", 8)),
            hidden_dim=int(getattr(self.cfg, "metric_fisher_hidden_dim", 128)),
            alpha=float(getattr(self.cfg, "metric_fisher_alpha", 0.25)),
            eps=float(getattr(self.cfg, "metric_fisher_eps", 1e-3)),
            shrinkage=float(getattr(self.cfg, "metric_fisher_shrinkage", 0.1)),
            relation_topk=int(getattr(self.cfg, "metric_fisher_relation_topk", 3)),
            init_scale=float(getattr(self.cfg, "metric_fisher_init_scale", 1e-3)),
            match_temperature=float(
                getattr(self.cfg, "metric_fisher_match_temperature", 1.0)
            ),
        ).cuda()
        self.learnable_fisher_matcher = matcher

        metric_params = [p for p in matcher.parameters() if p.requires_grad]
        if metric_params:
            base_lr = self.optimizer.param_groups[0]["lr"]
            fisher_lr = float(getattr(self.cfg, "metric_fisher_lr", base_lr))
            fisher_weight_decay = float(
                getattr(self.cfg, "metric_fisher_weight_decay", 0.0)
            )
            self.optimizer.add_param_group(
                {
                    "params": metric_params,
                    "lr": fisher_lr,
                    "initial_lr": fisher_lr,
                    "weight_decay": fisher_weight_decay,
                }
            )

        n_metric = sum(p.numel() for p in metric_params)
        self.logger.info(
            "=> Learnable covariance Fisher enabled for backbone representation learning: "
            f"params={n_metric}, feat_dim={matcher.feat_dim}, "
            f"spectral_rank={matcher.spectral_rank}, low_rank={matcher.low_rank}, "
            f"alpha={matcher.alpha}, eps={matcher.eps}, "
            f"grad_clip={getattr(self.cfg, 'metric_fisher_grad_clip', 5.0)}, "
            f"metric_compute=fp32, calibration_metric={self.semantic_metric_matcher.metric}"
        )

    def _model_module(self):
        return self.model.module if comm.get_world_size() > 1 else self.model

    def extract_3d_vlm_proto(self, regis_val_loader):
        """Build cosine VLM prototypes and backbone-space Fisher statistics.

        The VLM support features remain responsible only for the original cosine
        pseudo-label calibration. Fisher moments are estimated separately from
        the point-cloud backbone feature space so support and query features live
        in the same metric space.
        """
        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        vlm_feat_dim = 512
        fisher_feat_dim = int(getattr(self.cfg, "metric_fisher_feat_dim", 64))

        vlm_sums = torch.zeros(
            num_novel, vlm_feat_dim, device="cuda", dtype=torch.float32
        )
        vlm_counts = torch.zeros(num_novel, device="cuda", dtype=torch.long)

        fisher_sums = torch.zeros(
            num_novel, fisher_feat_dim, device="cuda", dtype=torch.float32
        )
        fisher_outers = torch.zeros(
            num_novel,
            fisher_feat_dim,
            fisher_feat_dim,
            device="cuda",
            dtype=torch.float32,
        )
        fisher_counts = torch.zeros(num_novel, device="cuda", dtype=torch.long)

        if getattr(self.semantic_metric_matcher, "needs_support_statistics", False):
            self.semantic_metric_matcher.reset_support_statistics(
                num_novel, vlm_feat_dim, torch.device("cuda")
            )

        model_module = self._model_module()
        was_training = model_module.training
        model_module.eval()

        try:
            for input_dict in regis_val_loader:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.cuda(non_blocking=True)

                with torch.no_grad():
                    # Frozen VLM features for the original cosine calibration.
                    data_dict = self.construct_3D_vlm_input(input_dict)
                    vlm_output = self.VLM_3D(data_dict)
                    vlm_feats = vlm_output["all_feats"].float()

                    # Trainable point-cloud backbone support features are sampled
                    # under no-grad. They define a fixed support geometry for this
                    # registration run, but are in the same space as train queries.
                    backbone_feats = model_module.getFeatures(input_dict).float()
                    if backbone_feats.shape[-1] != fisher_feat_dim:
                        raise RuntimeError(
                            "Fisher feature dimension mismatch: "
                            f"config={fisher_feat_dim}, backbone={backbone_feats.shape[-1]}"
                        )

                    for class_id in input_dict["segment"].unique():
                        class_id = int(class_id.item())
                        if class_id == -1:
                            continue
                        assert class_id >= self.cfg.data.num_bases
                        idx = class_id - self.cfg.data.num_bases
                        class_mask = input_dict["segment"] == class_id

                        class_vlm = vlm_feats[class_mask]
                        if class_vlm.numel() > 0:
                            vlm_sums[idx] += class_vlm.sum(dim=0)
                            vlm_counts[idx] += class_vlm.shape[0]
                            if getattr(
                                self.semantic_metric_matcher,
                                "needs_support_statistics",
                                False,
                            ):
                                self.semantic_metric_matcher.update_support_statistics(
                                    idx, class_vlm
                                )

                        class_backbone = backbone_feats[class_mask]
                        if class_backbone.numel() > 0:
                            fisher_sums[idx] += class_backbone.sum(dim=0)
                            fisher_outers[idx] += (
                                class_backbone.transpose(0, 1) @ class_backbone
                            )
                            fisher_counts[idx] += class_backbone.shape[0]
        finally:
            if was_training:
                model_module.train()

        if comm.get_world_size() > 1:
            # Every rank scans the same registration set. Broadcast rank-0
            # statistics so all ranks use identical support geometry.
            torch.distributed.broadcast(vlm_sums, src=0)
            torch.distributed.broadcast(vlm_counts, src=0)
            torch.distributed.broadcast(fisher_sums, src=0)
            torch.distributed.broadcast(fisher_outers, src=0)
            torch.distributed.broadcast(fisher_counts, src=0)

        if not torch.all(vlm_counts > 0):
            missing = torch.nonzero(vlm_counts <= 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Missing VLM support statistics for novel classes: {missing}")
        if not torch.all(fisher_counts > 0):
            missing = (
                torch.nonzero(fisher_counts <= 0, as_tuple=False).flatten().tolist()
            )
            raise RuntimeError(
                f"Missing backbone Fisher statistics for novel classes: {missing}"
            )

        if getattr(self.semantic_metric_matcher, "needs_support_statistics", False):
            self.semantic_metric_matcher.finalize_support_statistics()

        average_vlm_features = vlm_sums / vlm_counts.to(vlm_sums.dtype).unsqueeze(1)

        matcher = self.learnable_fisher_matcher
        if isinstance(matcher, LearnableCovarianceFisher):
            # Support geometry is intentionally detached: gradients should shape
            # query/backbone representations, not backpropagate through covariance
            # estimation/eigendecomposition on the few-shot support set.
            matcher.prepare_support_from_moments(
                fisher_sums.detach(),
                fisher_outers.detach(),
                fisher_counts.detach(),
            )
            diag = matcher.diagnostics()
            self.logger.info(
                "=> Backbone-space Fisher support statistics ready: "
                f"classes={num_novel}, counts={fisher_counts.tolist()}, "
                f"feat_dim={fisher_feat_dim}, rank={diag['spectral_rank']}, "
                f"A_rms={diag['a_rms']:.6f}, A_max={diag['a_max']:.6f}; "
                f"pseudo_label_metric={self.semantic_metric_matcher.metric}"
            )

        return average_vlm_features

    def pseudo_label_selection(
        self, all_pred_cloud, all_feat_cloud, offset, ps_thresh=0.6
    ):
        # Pseudo-label selection remains fully on the original semantic matcher
        # (cosine in this experiment); Fisher does not hard-filter labels.
        return super().pseudo_label_selection(
            all_pred_cloud,
            all_feat_cloud,
            offset,
            ps_thresh=ps_thresh,
        )

    def _sample_metric_points(self, features, labels):
        num_bases = self.cfg.data.num_bases
        num_all = self.cfg.data.num_base_novels
        valid = (labels >= num_bases) & (labels < num_all)
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        if idx.numel() == 0:
            return None, None

        max_points = int(getattr(self.cfg, "metric_fisher_max_points", 4096))
        if max_points > 0 and idx.numel() > max_points:
            perm = torch.randperm(idx.numel(), device=idx.device)[:max_points]
            idx = idx[perm]

        # IMPORTANT: no detach here. Fisher metric CE must backpropagate through
        # these point-wise features into the point-cloud backbone.
        query = features[idx]
        target = labels[idx].long() - num_bases
        return query, target

    def _compute_metric_loss(self, backbone_features, input_dict):
        matcher = self.learnable_fisher_matcher
        if not isinstance(matcher, LearnableCovarianceFisher):
            zero = backbone_features.new_zeros(())
            return zero, zero

        if backbone_features is None or matcher.support_stats is None:
            zero = self.vlm_novel_proto.new_zeros(())
            return zero, zero

        query, target = self._sample_metric_points(
            backbone_features, input_dict["segment"]
        )
        if query is None:
            zero = backbone_features.new_zeros(())
            return zero, zero

        if query.shape[-1] != matcher.feat_dim:
            raise RuntimeError(
                "Query/Fisher feature dimension mismatch: "
                f"query={query.shape[-1]}, fisher={matcher.feat_dim}"
            )

        loss_temp = max(
            float(getattr(self.cfg, "metric_fisher_loss_temperature", 1.0)),
            1e-6,
        )

        # Fisher whitening contains inverse variances, square roots and
        # quadratic forms. Keep the metric computation in fp32 even when the
        # segmentation forward uses AMP. The cast preserves autograd to the
        # original backbone feature tensor.
        with torch.cuda.amp.autocast(enabled=False):
            logits = matcher.logits(query.float(), temperature=loss_temp)
            metric_loss = F.cross_entropy(logits.float(), target)
            geo_reg = matcher.deformation_regularization().float()
        return metric_loss, geo_reg

    def _fisher_parameters_with_grad(self):
        matcher = self.learnable_fisher_matcher
        if matcher is None:
            return []
        return [
            p for p in matcher.parameters() if p.requires_grad and p.grad is not None
        ]

    def _sync_fisher_gradients(self):
        params = self._fisher_parameters_with_grad()
        if comm.get_world_size() <= 1 or not params:
            return
        world = float(comm.get_world_size())
        for param in params:
            torch.distributed.all_reduce(
                param.grad, op=torch.distributed.ReduceOp.SUM
            )
            param.grad.div_(world)

    def _fisher_gradients_finite(self):
        """Return one DDP-consistent finite/non-finite decision."""
        matcher = self.learnable_fisher_matcher
        if matcher is None:
            return True
        device = next(matcher.parameters()).device
        finite = torch.ones((), device=device, dtype=torch.int32)
        for param in self._fisher_parameters_with_grad():
            if not torch.isfinite(param.grad).all():
                finite.zero_()
                break
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        return bool(finite.item())

    def _clip_fisher_gradients(self):
        """Clip only finite Learnable Fisher-generator gradients after DDP averaging."""
        max_norm = float(getattr(self.cfg, "metric_fisher_grad_clip", 5.0))
        if max_norm <= 0:
            return None
        params = self._fisher_parameters_with_grad()
        if not params:
            return None
        return torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    def _handle_nonfinite_fisher_gradients(self):
        """Skip a contaminated optimizer step before NaNs reach parameters."""
        self.logger.warning(
            "=> Non-finite Learnable Fisher gradient detected after DDP sync; "
            "skipping this optimizer step."
        )
        self.optimizer.zero_grad(set_to_none=True)

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        # Pseudo-label selection and adaptive infilling remain cosine-based.
        new_target = self.target_calibrate(input_dict)

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            seg_logits = output_dict.pop("seg_logits")
            backbone_feat = output_dict.pop("backbone_feat")
            criteria = (
                self.model.module.criteria
                if comm.get_world_size() > 1
                else self.model.criteria
            )
            seg_loss = criteria(seg_logits, new_target)

            # _compute_metric_loss disables autocast internally. Crucially,
            # backbone_feat remains connected to the trainable backbone graph.
            metric_loss, geo_reg = self._compute_metric_loss(
                backbone_feat, input_dict
            )

            metric_weight = float(
                getattr(self.cfg, "metric_fisher_loss_weight", 0.05)
            )
            reg_weight = float(
                getattr(self.cfg, "metric_fisher_reg_weight", 1e-2)
            )
            loss = seg_loss + metric_weight * metric_loss + reg_weight * geo_reg
            output_dict["loss"] = loss
            output_dict["seg_loss"] = seg_loss.detach()
            output_dict["fisher_metric_loss"] = metric_loss.detach()
            output_dict["fisher_geo_reg"] = geo_reg.detach()

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            # The backbone is wrapped by DDP and synchronizes automatically.
            # The Fisher generator is trainer-side, so synchronize it manually.
            self.scaler.unscale_(self.optimizer)
            self._sync_fisher_gradients()
            if not self._fisher_gradients_finite():
                self._handle_nonfinite_fisher_gradients()
                new_scale = max(float(self.scaler.get_scale()) * 0.5, 1.0)
                self.scaler.update(new_scale=new_scale)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                self.comm_info["model_output_dict"] = output_dict
                return

            self._clip_fisher_gradients()
            self.scaler.step(self.optimizer)
            old_scale = self.scaler.get_scale()
            self.scaler.update()
            if old_scale <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self._sync_fisher_gradients()
            if not self._fisher_gradients_finite():
                self._handle_nonfinite_fisher_gradients()
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                self.comm_info["model_output_dict"] = output_dict
                return
            self._clip_fisher_gradients()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
