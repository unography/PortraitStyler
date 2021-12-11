import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2

from collections import namedtuple

import pytorch_lightning as pl
from models.u2net import U2NET_lite, U2NET_full
from datasets.arcanefaces import ArcaneFaces

class LitBase(pl.LightningModule):
    def __init__(self, cfg, model):
        super(LitBase, self).__init__()
        self.model = model
        self.cfg = cfg

        self.mae_loss = torch.nn.L1Loss()

    def prepare_data(self):
        self.train_dataset = ArcaneFaces(
            base_path=self.cfg["train_base_path"], mode="train", sz=320,rc=288)

    def forward(self, x):
        return self.model(x)

    def train_dataloader(self):
        
        # def collate_fn(batch):
        #     imgs, masks = [list(item) for item in zip(*batch)]
        #     size = [160, 192, 224, 256, 288][np.random.randint(0, 5)]
        #     w = size # imgs[0].size()[1]
        #     h = size # imgs[0].size()[2]
        #     len_imgs = len(imgs)
        #     tensor = torch.zeros((len_imgs, 3, h, w), dtype=torch.float32)
        #     targets = torch.zeros((len_imgs, 1, h, w), dtype=torch.float32)
        #     for i, img in enumerate(imgs):
        #         img = img.unsqueeze(0)
        #         out = F.interpolate(img, size=(w, h)) #img
        #         out = out.squeeze(0)
        #         mask = masks[i].unsqueeze(0)
        #         out_m = F.interpolate(mask, size=(w, h)) #img
        #         out_m = out_m.squeeze(0)
        #         tensor[i] += out
        #         targets[i] += out_m
        #     return tensor, targets

        # train_loader = torch.utils.data.DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=self.cfg['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.cfg["batch_size"], shuffle=True, drop_last=True, num_workers=self.cfg["num_workers"])
        return train_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.cfg["lr"],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
        # cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 645)
        return optimizer

    def multi_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        loss0 = self.mae_loss(d0, labels_v)
        loss1 = self.mae_loss(d1, labels_v)
        loss2 = self.mae_loss(d2, labels_v)
        loss3 = self.mae_loss(d3, labels_v)
        loss4 = self.mae_loss(d4, labels_v)
        loss5 = self.mae_loss(d5, labels_v)
        loss6 = self.mae_loss(d6, labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        d0, d1, d2, d3, d4, d5, d6 = self(inputs)
        loss2, loss = self.multi_loss_fusion(
            d0, d1, d2, d3, d4, d5, d6, labels)
        self.log('train_loss', loss)
        # wandb.log('train_loss', loss)
        return loss


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)



if __name__ == "__main__":

    pl.seed_everything(42)

    config = dict(
        train_base_path="/Drive/MyDrive/datasets/faces2comics",
        batch_size=16,
        epochs=200,
        lr=0.001,
        num_workers=2
    )

    u2net = U2NET_full()
    # u2net = U2NET_lite()
    pl_model = LitBase(config, u2net) 

    train_checkpoint_train_loss = pl.callbacks.ModelCheckpoint(
        dirpath=".",
        monitor="train_loss",
        filename="u2net_train_loss_{epoch:04d}_{train_loss:.2f}",
        mode="min"
    )
    
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, callbacks=[model_checkpoint], process_position=2)
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, callbacks=[model_checkpoint], process_position=2, precision=16, accelerator='ddp')
    # trainer = pl.Trainer(max_epochs=epochs, callbacks=[model_checkpoint], precision=16, amp_backend='apex', amp_level='O2', gpus=4)
    # trainer = pl.Trainer(
    #     max_epochs=config["epochs"],
    #     callbacks=[train_checkpoint_train_loss, CheckpointEveryNSteps(1200)],
    #     # precision=16,
    #     gpus=-1,
    #     accelerator='ddp',
    # )
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        callbacks=[train_checkpoint_train_loss, CheckpointEveryNSteps(1200)],
        # precision=16,
    )

    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', progress_bar_refresh_rate=100)
    # trainer = pl.Trainer(max_epochs=epochs, gpus=-1, deterministic=True, precision=16, callbacks=[model_checkpoint], accelerator='ddp', amp_backend='apex', amp_level='02')
    # trainer = pl.Trainer(max_epochs=epochs, deterministic=True, callbacks=[model_checkpoint], accelerator='ddp')

    trainer.fit(pl_model)