"""
Trainer for Step-1 CDC base-geometry pretraining.

It reuses the same frozen CLIP text encoder and prompt template as GFS-VL
registration, but does not build the 3D VLM and never sees novel few-shot data.
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
        super().__init__(cfg)
        self._build_and_inject_text_prior()

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _build_and_inject_text_prior(self):
        cfg_file = self.cfg.cdc_text_cfg_file
        self.logger.info(
            "=> Building frozen CLIP text prior for CDC base geometry ..."
        )
        cfg_from_yaml_file(cfg_file, pla_cfg)
        pla_cfg.CLASS_NAMES = list(self.cfg.data.names)
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

        model = self._unwrap_model()
        model.set_text_embeddings(text_embed.cuda().float())
        self.logger.info(
            "=> Injected frozen text prototypes: "
            f"{tuple(text_embed.shape)} for {len(self.cfg.data.names)} base classes"
        )

        del text_encoder
        torch.cuda.empty_cache()

    def before_epoch(self):
        model = self._unwrap_model()
        model.set_training_epoch(self.epoch)
        if self.epoch == model.text_warmup_epochs:
            self.logger.info(
                "=> CDC text-direction constraint is now enabled "
                f"(epoch {self.epoch + 1})."
            )
        super().before_epoch()
