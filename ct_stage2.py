from __future__ import print_function, division

import logging

import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.loss import l1_loss, ssim, psnr
from lib.background_generator import Background_Gen
from lib.mask_generator import UNet

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                       num_workers=self.cfg.batch_size * 2, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.bg_gen = Background_Gen(shape=(self.cfg.batch_size, 1024, 1024, 3), dataset_dir=dataset_dirs)
        # Mask predictor: input must be concattenated over batch, 3-channel output
        self.mask_gen = UNet(n_channels=3, n_classes=3)
        self.mask_optimizer = optim.Adam(self.mask_gen.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.mask_scheduler = optim.lr_scheduler.StepLR(self.mask_optimizer, step_size=4000, gamma=0.8)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        self.mask_gen.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        self.model.raft_stereo.freeze_bn()
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def train(self):

        le_reached = False
        sample_count = 10
        with tqdm(range(self.total_steps, self.cfg.num_steps)) as pbar:
            for _ in pbar:
                self.optimizer.zero_grad()
                self.mask_optimizer.zero_grad()
                data = self.fetch_data(phase='train')

                # Raft Stereo + GS Regresser
                data, flow_loss, metrics = self.model(data, is_train=True)
                # Gaussian Render
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()

                bgs1 = self.bg_gen.generate_background().cuda()

                comps_l, masks_l = self.bg_gen.create_composites(images=data['lmain']['img'], masks=data['lmain']['mask'], backgrounds=bgs1, random_flip=True)
                comps_r, masks_r = self.bg_gen.create_composites(images=data['rmain']['img'], masks=data['rmain']['mask'], backgrounds=bgs1, random_flip=True)

                # Concatenate composites along batch dimension
                composites = torch.cat([comps_l, comps_r], dim=0).cuda()

                # Forward pass through the mask generator
                logits = self.mask_gen(composites)
                ground_truth_left = masks_l
                ground_truth_right = masks_r
                ground_truth = torch.cat([ground_truth_left, ground_truth_right], dim=0).cuda()

                # Losses for the mask generator
                total_segmentation_loss = self.criterion(logits, ground_truth)

                total_segmentation_loss.backward()
                self.mask_optimizer.step()
                self.mask_scheduler.step()
                # Sampling for high loss
                if total_segmentation_loss < 0.004:
                    le_reached = True
                if le_reached and total_segmentation_loss>0.1 and sample_count>0:
                    composites = ((composites + 1) * 255 / 2).cpu().numpy().astype(np.uint8)
                    # Convert to (H, W, C) format and concatenate
                    composites = np.transpose(composites, (0, 2, 3, 1))

                    composite_image = np.concatenate(composites, axis=1)
                    composite_name = '%s/%s_high_err_sample.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(composite_name, composite_image[:, :, ::-1])
                    sample_count -=1

                Ll1 = l1_loss(render_novel, gt_novel)
                Lssim = 1.0 - ssim(render_novel, gt_novel)
                loss = 1.0 * flow_loss + 0.8 * Ll1 + 0.2 * Lssim

                if (self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0) or self.total_steps in [500, 2000, 4000, 5000]:
                    self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                    self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
                metrics.update({
                    'l1': Ll1.item(),
                    'ssim': Lssim.item(),
                    'segmentation_loss': total_segmentation_loss.item()
                })
                self.logger.push(metrics)

                pbar.set_postfix({
                    'GPS Loss': loss.item(),
                    'Segmentation Loss': total_segmentation_loss.item()
                })

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()

                if (self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0) or self.total_steps in [500, 2000, 4000, 5000]:
                    self.model.eval()
                    self.run_eval()
                    self.model.train()
                    self.model.raft_stereo.freeze_bn()

                self.total_steps += 1

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))


    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list, psnr_list, segmentation_loss_list = [], [], [], []
        for idx in range(int(self.len_val * 0.2)):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                bgs1 = self.bg_gen.generate_background().cuda()
                comps_l, _ = self.bg_gen.create_composites(images=data['lmain']['img'], masks=data['lmain']['mask'], backgrounds=bgs1)
                comps_r, _ = self.bg_gen.create_composites(images=data['rmain']['img'], masks=data['rmain']['mask'], backgrounds=bgs1)
                composites = torch.cat([comps_l, comps_r], dim=0).cuda()

                logits = self.mask_gen(composites)
                ground_truth_left = data['lmain']['mask']
                ground_truth_right = data['rmain']['mask']
                ground_truth = torch.cat([ground_truth_left, ground_truth_right], dim=0).cuda()

                total_segmentation_loss = self.criterion(logits, ground_truth)
                segmentation_loss_list.append(total_segmentation_loss.item())

                if idx == 1:
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

                    for view in ['lmain', 'rmain']:
                        valid = (data[view]['valid'] >= 0.5)
                        epe = torch.sum((data[view]['flow'] - data[view]['flow_pred']) ** 2, dim=1).sqrt()
                        epe = epe.view(-1)[valid.view(-1)]
                        one_pix = (epe < 1)
                        epe_list.append(epe.mean().item())
                        one_pix_list.append(one_pix.float().mean().item())

                    # Raw mask
                    tmp_mask = logits[0].detach().sigmoid()
                    tmp_mask_raw = tmp_mask.permute(1, 2, 0).cpu().numpy() * 255

                    # Thresholded mask
                    threshold = 0.5
                    tmp_mask_thresholded = (tmp_mask > threshold).float() * 255
                    tmp_mask_thresholded = tmp_mask_thresholded.permute(1, 2, 0).cpu().numpy()

                    # Corresponding view
                    tmp_image = data['lmain']['img'][0].detach().permute(1, 2, 0).cpu().numpy()
                    tmp_image = ((tmp_image + 1) * 127.5).astype(np.uint8)

                    tmp_mask_raw = tmp_mask_raw.astype(np.uint8)
                    tmp_mask_thresholded = tmp_mask_thresholded.astype(np.uint8)

                    concat_image = np.concatenate((tmp_image[:, :, ::-1], tmp_mask_raw, tmp_mask_thresholded), axis=1)

                    tmp_concat_name = '%s/%s_gt_masks.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_concat_name, concat_image)

                    # Save composites
                    # De-norm
                    composites_denorm = ((composites.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    composite_concat = np.concatenate([composites_denorm[0], composites_denorm[1]], axis=1)
                    composite_name = '%s/%s_composites.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(composite_name, composite_concat[:, :, ::-1])

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_pix = np.round(np.mean(np.array(one_pix_list)), 4)
        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_segmentation_loss = np.round(np.mean(np.array(segmentation_loss_list)), 4)
        logging.info(f"Validation Metrics ({self.total_steps}): epe {val_epe}, 1pix {val_one_pix}, psnr {val_psnr}, seg_loss {val_segmentation_loss}")
        self.logger.write_dict({'val_epe': val_epe, 'val_1pix': val_one_pix, 'val_psnr': val_psnr, 'val_segmentation_loss': val_segmentation_loss}, write_step=self.total_steps)
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")

        # Load Segmenter
        self.mask_gen.load_state_dict(ckpt['mask_network'], strict=strict)
        self.mask_optimizer.load_state_dict(ckpt['mask_optimizer'])
        self.mask_scheduler.load_state_dict(ckpt['mask_scheduler'])
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Network loading complete")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Saving checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'mask_network': self.mask_gen.state_dict(),
            'mask_optimizer': self.mask_optimizer.state_dict(),
            'mask_scheduler': self.mask_scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/stage2.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    bg_categories = ['brick', 'carpet', 'fabric', 'foliage', 'metal', 'tile', 'wallpaper', 'wood']
    base_path = '/data/scratch/ec23984/data-repo/minc/minc-2500/images'
    dataset_dirs = [os.path.join(base_path, label) for label in bg_categories]

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
