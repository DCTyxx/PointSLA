"""
Trainer

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
import numpy as np
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
from .semantic_metric import SemanticMetricMatcher
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

from pcseg.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
)
from pathlib import Path
from pointcept.models.PLA.pcseg.models.text_networks import (
    load_text_embedding_from_encoder,
    load_text_embedding_from_path,
)
from pointcept.models.PLA.pcseg.models import (
    build_vision_network,
    build_text_network,
)
import torch.distributed as dist

TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(
                ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
            )
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = (
            SummaryWriter(self.cfg.save_path)
            if comm.is_main_process()
            else None
        )
        self.logger.info(
            f"Tensorboard writer logging dir: {self.cfg.save_path}"
        )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data
            )
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=(
                False if self.cfg.num_worker_per_gpu == 0 else True
            ),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data
                )
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(
            self.cfg.optimizer, self.model, self.cfg.param_dicts
        )

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader) * self.cfg.eval_epoch
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("GFS_VL_Trainer")
class GFS_VL_Trainer(Trainer):
    """
    Trainer for generalized few-shot 3D point cloud segmentation.
    """

    def __init__(self, cfg):
        """Initialize the trainer and build the 3D VLM."""
        super().__init__(cfg)
        self.build_zero_shot_model()

    def build_zero_shot_model(self):
        """
        Build the 3D VLM model used for pseudo-label generation.
        """
        cfg_file = "./pointcept/models/PLA/tools/cfgs/scannet200_models/zs/spconv_clip_caption_openscene.yaml"
        self.VLM_3D_cfg = cfg_from_yaml_file(cfg_file, cfg)

        # Combine base and novel class names for the 3D VLM model
        cfg.CLASS_NAMES = (
            self.train_loader.dataset.base_class_names
            + self.train_loader.dataset.novel_class_names
        )
        self.num_class = self.cfg.data.num_base_novels

        # Configure experiment tag and group path
        cfg.TAG = Path(cfg_file).stem
        cfg.EXP_GROUP_PATH = "/".join(cfg_file.split("/")[1:-1])

        # Update configuration for text embedding extraction
        cfg_from_list(["TEXT_ENCODER.EXTRACT_EMBED", True], cfg)

        logger = self.logger
        # Build text embedding if specified
        if cfg.get("TEXT_ENCODER", None) or cfg.MODEL.TASK_HEAD.get(
            "TEXT_EMBED", None
        ):
            text_encoder = build_text_network(cfg.TEXT_ENCODER).cuda()
            if (
                cfg.get("TEXT_ENCODER", None)
                and cfg.TEXT_ENCODER.EXTRACT_EMBED
            ):
                text_embed = load_text_embedding_from_encoder(
                    cfg.TEXT_ENCODER, text_encoder, logger
                )
            else:
                text_embed = load_text_embedding_from_path(
                    cfg.MODEL.TASK_HEAD.TEXT_EMBED, logger
                )
            cfg.MODEL.TASK_HEAD.TEXT_EMBED.CHANNEL = text_embed.shape[1]
            cfg.MODEL.TASK_HEAD.TEXT_EMBED.NUM_CLASS = text_embed.shape[0]
            if cfg.MODEL.get("ADAPTER", False):
                cfg.MODEL.ADAPTER.TEXT_DIM = text_embed.shape[1]
        else:
            text_embed = None

        # Build the 3D VLM model
        self.VLM_3D = build_vision_network(
            model_cfg=cfg.MODEL,
            num_class=cfg.CLASS_NAMES,
            dataset=cfg.DATA_CONFIG,
        )
        self.VLM_3D.load_params_from_file(
            filename=self.cfg.vlm_3d_weight,
            logger=logger,
            epoch_id=None,
            to_cpu=False,
        )

        # Set the classification head with text embeddings if available
        if text_embed is not None:
            self.VLM_3D.task_head.set_cls_head_with_text_embed(text_embed)

        # Unified semantic metric used by pseudo-label selection and adaptive
        # infilling. The default "cosine" exactly recovers the original GFS-VL
        # matching rule; the other options only change the feature-space metric.
        self.semantic_metric_matcher = SemanticMetricMatcher(
            metric=getattr(self.cfg, "semantic_metric", "cosine"),
            text_embed=text_embed,
            pca_rank=getattr(self.cfg, "metric_pca_rank", 32),
            pca_shrinkage=getattr(self.cfg, "metric_pca_shrinkage", 0.1),
            metric_ridge=getattr(self.cfg, "metric_ridge", 1e-3),
            fisher_temperature=getattr(
                self.cfg, "metric_fisher_temperature", 1.0
            ),
        )
        self.logger.info(
            "=> Semantic matching metric: "
            f"{self.semantic_metric_matcher.metric}"
        )

        self.VLM_3D.cuda()
        self.VLM_3D.eval()
        # Freeze parameters of the 3D VLM model
        for param in self.VLM_3D.parameters():
            param.requires_grad = False

    def extract_3d_vlm_proto(self, regis_val_loader):
        """
        Compute novel-class prototypes using the frozen 3D VLM and, when
        requested, fit support-set PCA statistics for metric ablations.
        """
        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        feat_dim = 512
        average_features = torch.zeros(num_novel, feat_dim, device="cuda")
        class_counts = torch.zeros(num_novel, device="cuda")

        if self.semantic_metric_matcher.needs_support_statistics:
            self.semantic_metric_matcher.reset_support_statistics(
                num_classes=num_novel,
                feat_dim=feat_dim,
                device=average_features.device,
            )

        # Each process computes the same registration statistics. This keeps
        # metric state available locally under DDP without serializing large
        # covariance/factor tensors between ranks.
        for input_dict in regis_val_loader:
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            with torch.no_grad():
                data_dict = self.construct_3D_vlm_input(input_dict)
                output_dict = self.VLM_3D(data_dict)
                vlm_feats = output_dict["all_feats"]

                for class_id in input_dict["segment"].unique():
                    class_id = class_id.item()
                    if class_id == -1:
                        continue
                    assert class_id >= self.cfg.data.num_bases
                    class_mask = input_dict["segment"] == class_id
                    class_features = vlm_feats[class_mask]
                    idx = class_id - self.cfg.data.num_bases
                    average_features[idx] += class_features.sum(dim=0)
                    class_counts[idx] += class_features.size(0)
                    self.semantic_metric_matcher.update_support_statistics(
                        idx, class_features
                    )

        assert torch.all(class_counts > 0)
        average_features = average_features / class_counts.unsqueeze(1)

        if self.semantic_metric_matcher.needs_support_statistics:
            self.semantic_metric_matcher.finalize_support_statistics()

        if comm.get_world_size() > 1:
            dist.broadcast(average_features, src=0)
        return average_features

    def build_model(self):
        """
        Construct the model, convert it for DDP if needed,
        and load pre-trained weights.
        """
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")

        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        # Load pre-trained weights if available
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # Remove 'module.' for single GPU
                else:
                    if comm.get_world_size() > 1:
                        key = (
                            "module." + key
                        )  # Prepend 'module.' for multi-GPU
                weight[key] = value
            strict = (
                self.cfg.load_strict if "load_strict" in self.cfg else True
            )
            missing_keys, unexpected_keys = model.load_state_dict(
                weight, strict=strict
            )
            self.logger.info(
                f"=> Loaded weight '{self.cfg.weight}' (epoch {checkpoint['epoch']})"
            )
            self.logger.info(
                f"=> missing_keys {missing_keys}, unexpected_keys {unexpected_keys}"
            )

        return model

    def train(self):
        """
        Main training loop for generalized few-shot learning. Iterates over multiple
        registration datasets and trains the segmentation model accordingly.
        """
        for self.regis_str in self.cfg.regis_train_list:
            # Retrieve registration dataset configuration
            if hasattr(self.cfg.data, self.regis_str):
                val_data_cfg = getattr(self.cfg.data, self.regis_str)
            else:
                raise AttributeError(
                    f"{self.regis_str} not found in self.cfg.data"
                )
            val_data = build_dataset(val_data_cfg)
            regis_val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=None,
                collate_fn=collate_fn,
            )

            # Extract 3D VLM feature prototypes on novel classes from current registration set
            self.vlm_novel_proto = self.extract_3d_vlm_proto(regis_val_loader)

            # Update the training dataset configuration
            self.cfg.data.train.k_shot = self.cfg.k_shot
            self.cfg.data.train.seed = val_data_cfg["seed"]
            self.cfg.data.train.nb_mix_blks = self.cfg.nb_mix_blks
            self.train_loader = self.build_train_loader()

            # Execute the training loop for the current registration dataset
            with EventStorage() as self.storage, ExceptionWriter():
                self.before_train()
                self.logger.info(
                    ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
                )
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        self.before_step()
                        self.run_step()
                        self.after_step()
                    self.after_epoch()
                self.after_train()

        # If using multiple registration datasets, log the final evaluation metrics
        if comm.is_main_process() and len(self.cfg.regis_train_list) > 1:
            class_names = (
                self.train_loader.dataset.base_class_names
                + self.train_loader.dataset.novel_class_names
            )
            mean_mIoUs = self.hooks[-1].mean_mIoUs / len(
                self.hooks[-1].mean_iou_list
            )
            mIoU_val, base_mIoU, novel_mIoU, hm_mIoU = mean_mIoUs
            self.logger.info(
                f"Final Eval result from {self.cfg.regis_train_list}: mIoU: {mIoU_val:.4f}, "
                f"BASE: {base_mIoU:.4f}, NOVEL: {novel_mIoU:.4f}, hm_mIoU: {hm_mIoU:.4f}"
            )
            # Log class-wise IoU results
            stack_iou = np.mean(
                np.stack(self.hooks[-1].mean_iou_list, axis=0), axis=0
            )
            for i, iou in enumerate(stack_iou):
                self.logger.info(
                    f"Class_{i} - {class_names[i]} Final Result: iou {iou:.4f}"
                )

    def run_step(self):
        """
        Execute a single training step:
          - Calibrate target labels using the 3D VLM.
          - Forward pass through the segmentation model.
          - Compute loss and update model parameters.
        """
        input_dict = self.comm_info["input_dict"]
        # Move tensor inputs to GPU
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        # Calibrate target labels based on 3D VLM predictions
        new_target = self.target_calibrate(input_dict)

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            seg_logits = output_dict.pop("seg_logits")
            criteria = (
                self.model.module.criteria
                if comm.get_world_size() > 1
                else self.model.criteria
            )
            loss = criteria(seg_logits, new_target)
            output_dict["loss"] = loss

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            # Adjust the scheduler if loss scaling factor remains unchanged
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def pseudo_label_selection(
        self, all_pred_cloud, all_feat_cloud, offset, ps_thresh=0.6
    ):
        """
        Filter raw 3D VLM predictions based on prototype similarity.

        Args:
            all_pred_cloud (Tensor): Raw 3D VLM predictions for all points.
            all_feat_cloud (Tensor): Feature vectors for all points.
            offset (list): Batch offsets indicating point indices.
            ps_thresh (float): Cosine similarity threshold for filtering.

        Returns:
            Tensor: Filtered 3D VLM predictions.
        """
        filtered_preds = []
        # Process each batch using provided offsets
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            class_prototypes = {}
            pred_cloud = all_pred_cloud[start_idx:end_idx].clone()
            feat_cloud = all_feat_cloud[start_idx:end_idx].clone()

            # Calculate mean feature for each predicted class in the batch
            for class_id in pred_cloud.unique():
                class_mask = pred_cloud == class_id
                class_prototypes[class_id.item()] = feat_cloud[
                    class_mask
                ].mean(dim=0)

            # Filter predictions based on similarity with 3D VLM prototypes
            for class_id, proto in class_prototypes.items():
                if class_id < self.cfg.data.num_bases:
                    # Ignore base classes
                    pred_cloud[pred_cloud == class_id] = -1
                else:
                    novel_proto = self.vlm_novel_proto[
                        class_id - self.cfg.data.num_bases
                    ]
                    novel_idx = class_id - self.cfg.data.num_bases
                    similarity = self.semantic_metric_matcher.similarity(
                        proto,
                        novel_proto,
                        class_indices=[novel_idx],
                    ).squeeze()
                    if similarity < ps_thresh:
                        self.logger.info(
                            f"=> Filtering class {class_id}, "
                            f"metric={self.semantic_metric_matcher.metric}, "
                            f"similarity {similarity}"
                        )
                        pred_cloud[pred_cloud == class_id] = -1

            filtered_preds.append(pred_cloud)

        return torch.cat(filtered_preds, dim=0)

    @torch.no_grad()
    def target_calibrate(self, input_dict):
        """
        Calibrate the segmentation target labels using 3D VLM and few-shot data.
        """
        data_dict = self.construct_3D_vlm_input(input_dict)
        ret_dict = self.VLM_3D(data_dict)

        label = input_dict["segment"].clone()
        bg_label_mask = label == -1  # Default background label

        prediction = ret_dict["seg_preds"].clone()
        # Filter predictions using prototype similarity
        prediction = self.pseudo_label_selection(
            prediction,
            ret_dict["all_feats"],
            input_dict["offset"].tolist(),
            ps_thresh=self.cfg.ps_thresh,
        )

        # Adaptive Infilling to further refine the pseudo-labels
        offset = input_dict["offset"].tolist()
        new_predictions = []
        # Process each batch of points
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            current_prototypes = {}
            pred_cloud = prediction[start_idx:end_idx].clone()
            feat_cloud = ret_dict["all_feats"][start_idx:end_idx].clone()
            cur_label = label[start_idx:end_idx].clone()
            cur_bg_mask = bg_label_mask[start_idx:end_idx]
            bg_pred_cloud = pred_cloud[cur_bg_mask]
            bg_feat_cloud = feat_cloud[cur_bg_mask]

            if "mask" in input_dict:
                noda_mask = input_dict["mask"][start_idx:end_idx].clone()

            # Compute prototypes for each novel class present in the background region
            for class_id in bg_pred_cloud.unique():
                if class_id < 0:
                    continue
                assert class_id >= self.cfg.data.num_bases
                class_features = bg_feat_cloud[bg_pred_cloud == class_id]
                current_prototypes[class_id.item()] = class_features.mean(
                    dim=0
                )

            updated_prototypes = []
            updated_class_ids = []
            # Prepare updated prototypes for all novel classes
            for class_id in range(
                self.cfg.data.num_bases, self.cfg.data.num_base_novels
            ):
                if class_id in current_prototypes:
                    updated_prototypes.append(current_prototypes[class_id])
                else:
                    updated_prototypes.append(
                        self.vlm_novel_proto[
                            class_id - self.cfg.data.num_bases
                        ].clone()
                    )
                updated_class_ids.append(class_id)
            updated_prototypes = torch.stack(updated_prototypes)

            # Identify points with unassigned labels (-1) for refinement
            if "mask" in input_dict:
                mask_neg_one = (
                    (pred_cloud == -1) & (cur_label == -1) & (noda_mask == 1)
                )
            else:
                mask_neg_one = (pred_cloud == -1) & (cur_label == -1)
            feat_neg_one = feat_cloud[mask_neg_one]

            # Match unlabeled points to updated novel prototypes using the
            # same metric as pseudo-label selection.
            metric_class_indices = list(range(len(updated_class_ids)))
            similarities = self.semantic_metric_matcher.similarity(
                feat_neg_one,
                updated_prototypes,
                class_indices=metric_class_indices,
            )
            max_similarity, max_class_idx = similarities.max(dim=1)
            refined_labels = pred_cloud.clone()
            refined_labels[mask_neg_one] = torch.where(
                max_similarity >= self.cfg.ai_thresh,
                torch.tensor(updated_class_ids, device=pred_cloud.device)[
                    max_class_idx
                ],
                -1,  # Retain -1 if below threshold
            )
            new_predictions.append(refined_labels)

        # Merge predictions from all batches and update background regions in the original label
        prediction = torch.cat(new_predictions, dim=0)
        label_one_hot = label.clone()
        label_one_hot[bg_label_mask] = prediction[bg_label_mask]

        return label_one_hot

    def construct_3D_vlm_input(self, input_dict):
        """
        Construct the input dictionary required for 3D VLM.
        """
        xyz = input_dict["coord"]
        rgb = input_dict["color"]  # Colors normalized to [-1, 1]

        # Adjust RGB if not normalized
        if not self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.rgb_norm:
            rgb = (rgb + 1) * 127.5

        data_dict = {
            "points_xyz": xyz,
            "rgb": rgb,
            "pc_count": xyz.shape[0],
        }
        # Scale coordinates for voxelization
        xyz_voxel_scale = (
            xyz * self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.voxel_scale
        )
        xyz_voxel_scale = xyz_voxel_scale - xyz_voxel_scale.min(0).values
        data_dict["points_xyz_voxel_scale"] = xyz_voxel_scale

        # Prepare features: use RGB and/or coordinates as needed
        if self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.rgb_as_feat:
            data_dict["feats"] = data_dict["rgb"]

        if self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.xyz_as_feat:
            if "feats" in data_dict:
                data_dict["feats"] = torch.cat(
                    (data_dict["feats"], data_dict["points_xyz"]), dim=1
                )
            else:
                data_dict["feats"] = data_dict["points_xyz"]

        # Process offsets to generate indexed voxel coordinates
        result = []
        start = 0
        for i, end in enumerate(input_dict["offset"]):
            slice_xyz = data_dict["points_xyz_voxel_scale"][start:end]
            # Prepend batch index to each point coordinate
            indexed_points = torch.cat(
                [
                    torch.full(
                        (slice_xyz.shape[0], 1),
                        i,
                        dtype=torch.int64,
                        device=slice_xyz.device,
                    ),
                    slice_xyz.to(torch.int64),
                ],
                dim=-1,
            )
            result.append(indexed_points)
            start = end

        data_dict["points_xyz_voxel_scale"] = torch.cat(result, dim=0)

        # Determine the spatial shape for the voxel grid
        data_dict["spatial_shape"] = torch.clamp(
            (data_dict["points_xyz_voxel_scale"].max(0).values[1:] + 1),
            min=self.VLM_3D_cfg.DATA_CONFIG.MIN_SPATIAL_SCALE,
        )
        data_dict["batch_idxs"] = data_dict["points_xyz_voxel_scale"][:, 0].to(
            torch.int32
        )

        batch_size = len(input_dict["offset"])
        if batch_size == 1:
            data_dict["offsets"] = torch.tensor(
                [0, data_dict["batch_idxs"].shape[0]], dtype=torch.int32
            )
        else:
            data_dict["offsets"] = torch.cumsum(
                torch.bincount(data_dict["batch_idxs"] + 1).to(torch.int32),
                dim=0,
            )
            assert len(data_dict["offsets"]) == batch_size + 1

        data_dict["batch_size"] = batch_size
        return data_dict
