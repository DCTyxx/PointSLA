import torch

import pointcept.utils.comm as comm
from .train import GFS_VL_Trainer, TRAINERS


@TRAINERS.register_module("GFS_VL_RMM_CDC_Trainer")
class GFS_VL_RMM_CDC_Trainer(GFS_VL_Trainer):
    """GFS-VL trainer with conditional-CDC RMM geometry learning."""

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        # Keep the baseline pseudo-label construction and cosine semantic
        # matching unchanged.  RMM acts on the trainable PTv3 representation.
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

            cdc_loss = output_dict.get("rmm_cdc_loss", None)
            if cdc_loss is None:
                raise RuntimeError(
                    "RegisTrainRMMCDCSegmentor must return rmm_cdc_loss in training mode"
                )
            cdc_weight = float(getattr(self.cfg, "rmm_cdc_loss_weight", 0.01))
            loss = seg_loss + cdc_weight * cdc_loss
            output_dict["seg_loss"] = seg_loss
            output_dict["rmm_cdc_loss"] = cdc_loss
            output_dict["loss"] = loss

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            scale_before = self.scaler.get_scale()
            self.scaler.update()
            if scale_before <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict
