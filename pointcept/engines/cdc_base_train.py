"""
Trainer for Step-1 CDC base-geometry pretraining.

It reuses the frozen CLIP text encoder and prompt template from GFS-VL, but it
strictly restricts Step 1 to base-class data and base-class names. No novel
few-shot samples, registration sets, or novel class names are allowed here.
"""

import torch

from pcseg.config import cfg as pla_cfg
from pcseg.config import cfg_from_list, cfg_from_yaml_file
from pointcept.models.PLA.pcseg.models import build_text_network
from pointcept.models.PLA.pcseg.models.text_networks import (
    load_text_embedding_from_encoder,
)

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
