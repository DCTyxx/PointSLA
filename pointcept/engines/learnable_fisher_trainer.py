import torch
import torch.nn.functional as F

from .learnable_fisher import LearnableFisherRiemannianMatcher
from .train import GFS_VL_Trainer, TRAINERS
import pointcept.utils.comm as comm


@TRAINERS.register_module("GFS_VL_LearnableFisher_Trainer")
class GFS_VL_LearnableFisher_Trainer(GFS_VL_Trainer):
    """GFS-VL trainer with a task-learnable Fisher-Riemannian matcher.

    The 3D VLM remains frozen. Only a lightweight semantic-space deformation
    of the Fisher pullback metric is learned jointly with the original
    segmentation model. The metric head is training-only: it changes
    pseudo-label calibration and receives its own task loss, while the final
    segmentation network and inference path remain unchanged.
    """

    def build_zero_shot_model(self):
        super().build_zero_shot_model()

        enabled = bool(getattr(self.cfg, "metric_fisher_learnable", False))
        if self.semantic_metric_matcher.metric != "fisher_riemannian" or not enabled:
            return

        frozen_text = self.semantic_metric_matcher.text_embed
        matcher = LearnableFisherRiemannianMatcher(
            text_embed=frozen_text,
            metric_ridge=getattr(self.cfg, "metric_ridge", 1e-3),
            fisher_temperature=getattr(
                self.cfg, "metric_fisher_temperature", 1.0
            ),
            hidden_dim=getattr(self.cfg, "metric_fisher_hidden_dim", 32),
            residual_scale=getattr(
                self.cfg, "metric_fisher_residual_scale", 0.25
            ),
            learn_temperature=getattr(
                self.cfg, "metric_fisher_learn_temperature", True
            ),
            max_log_weight=getattr(
                self.cfg, "metric_fisher_max_log_weight", 2.0
            ),
        ).cuda()
        self.semantic_metric_matcher = matcher

        # The trainer is constructed after the segmentation optimizer, so add
        # only the Fisher parameters as a dedicated optimization group.  The
        # matcher is intentionally not attached to the segmentation model:
        # inference therefore keeps the exact original GFS-VL architecture.
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

        self.logger.info(
            "=> Learnable Fisher enabled: "
            f"{matcher.parameter_summary()}, "
            f"loss_weight={getattr(self.cfg, 'metric_fisher_loss_weight', 0.1)}"
        )

    def pseudo_label_selection(
        self, all_pred_cloud, all_feat_cloud, offset, ps_thresh=0.6
    ):
        # target_calibrate() executes under torch.no_grad(). Cache the frozen
        # VLM features here and train only the Fisher matcher afterwards.
        self._metric_train_features = all_feat_cloud.detach()
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

        query = features[idx].detach()
        target = labels[idx].long() - num_bases
        return query, target

    def _compute_metric_loss(self, input_dict):
        if not isinstance(
            self.semantic_metric_matcher, LearnableFisherRiemannianMatcher
        ):
            zero = self.vlm_novel_proto.new_zeros(())
            return zero, zero

        features = getattr(self, "_metric_train_features", None)
        if features is None:
            zero = self.vlm_novel_proto.new_zeros(())
            return zero, zero

        query, target = self._sample_metric_points(
            features, input_dict["segment"]
        )
        if query is None:
            zero = self.vlm_novel_proto.new_zeros(())
            return zero, zero

        similarities = self.semantic_metric_matcher.similarity(
            query, self.vlm_novel_proto
        )
        loss_temp = max(
            float(getattr(self.cfg, "metric_fisher_loss_temperature", 0.1)),
            1e-4,
        )
        metric_loss = F.cross_entropy(similarities / loss_temp, target)
        geo_reg = self.semantic_metric_matcher.deformation_regularization(query)
        return metric_loss, geo_reg

    def _sync_fisher_gradients(self):
        if comm.get_world_size() <= 1:
            return
        world = float(comm.get_world_size())
        for param in self.semantic_metric_matcher.parameters():
            if param.requires_grad and param.grad is not None:
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM
                )
                param.grad.div_(world)

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        # The original hard pseudo-label path stays unchanged and uses the
        # current Fisher geometry. Frozen VLM features are cached by the
        # overridden pseudo_label_selection().
        new_target = self.target_calibrate(input_dict)

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            seg_logits = output_dict.pop("seg_logits")
            criteria = (
                self.model.module.criteria
                if comm.get_world_size() > 1
                else self.model.criteria
            )
            seg_loss = criteria(seg_logits, new_target)

            metric_loss, geo_reg = self._compute_metric_loss(input_dict)
            metric_weight = float(
                getattr(self.cfg, "metric_fisher_loss_weight", 0.1)
            )
            reg_weight = float(
                getattr(self.cfg, "metric_fisher_reg_weight", 1e-3)
            )
            loss = seg_loss + metric_weight * metric_loss + reg_weight * geo_reg
            output_dict["loss"] = loss
            output_dict["seg_loss"] = seg_loss.detach()
            output_dict["fisher_metric_loss"] = metric_loss.detach()
            output_dict["fisher_geo_reg"] = geo_reg.detach()

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self._sync_fisher_gradients()
            self.scaler.step(self.optimizer)
            old_scale = self.scaler.get_scale()
            self.scaler.update()
            if old_scale <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self._sync_fisher_gradients()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
