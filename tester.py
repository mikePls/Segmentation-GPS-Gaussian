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
from lib.GaussianRender import pts2render
from lib.loss import l1_loss, ssim, psnr
from lib.background_generator import Background_Gen
from lib.mask_generator import UNet
import random

import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm


class Tester:
    def __init__(self, cfg_file):
        self.cfg = cfg_file

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)

        self.bg_gen = Background_Gen(shape=(self.cfg.batch_size, 1024, 1024, 3), dataset_dir=dataset_dirs, dataset_only=True) #dataset_dir=dataset_dirs,
        self.mask_gen = UNet(n_channels=3, n_classes=3)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.model.cuda()
        self.mask_gen.cuda()

        print(f"Using pre-trained model from: self.cfg.restore_ckpt")
        self.load_ckpt(self.cfg.restore_ckpt)

        self.model.eval()

    def run_test(self):
        logging.info(f"Starting testing ...")
        torch.cuda.empty_cache()
        epe_list, one_pix_list, psnr_list, segmentation_loss_list = [], [], [], []

        random_sample_id = random.sample(range(1, self.len_val), 5) # Save 5 random samples
        for idx in tqdm(range(self.len_val), desc="Testing Progress"):
            data = self.fetch_data()
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

                if idx in random_sample_id:  # Save 5 random samples
                    tmp_novel = data['novel_view']['img_pred'][0].detach() * 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_novel = cv2.resize(tmp_novel, (1024, 1024))

                    # Raw mask
                    tmp_mask = logits[0].detach().sigmoid()
                    tmp_mask_raw = tmp_mask.permute(1, 2, 0).cpu().numpy() * 255
                    tmp_mask_raw = tmp_mask_raw.astype(np.uint8)

                    # Thresholded mask
                    threshold = 0.5
                    tmp_mask_thresholded = (tmp_mask > threshold).float() * 255
                    tmp_mask_thresholded = tmp_mask_thresholded.permute(1, 2, 0).cpu().numpy()
                    tmp_mask_thresholded = tmp_mask_thresholded.astype(np.uint8)

                    # Composite image
                    tmp_composite = comps_l[0].detach()
                    tmp_composite = tmp_composite.permute(1, 2, 0).cpu().numpy()
                    tmp_composite = (tmp_composite + 1) * 127.5  # Scale to [0, 255]
                    tmp_composite = tmp_composite.astype(np.uint8)

                    # Concat horizontally
                    combined_image = np.hstack((tmp_mask_thresholded, tmp_mask_raw, tmp_composite))

                    # Save the combined image
                    tmp_img_name = f'{self.cfg.record.show_path}/sample_{idx}.jpg'
                    cv2.imwrite(tmp_img_name, combined_image[:, :, ::-1].astype(np.uint8))


                    for view in ['lmain', 'rmain']:
                        valid = (data[view]['valid'] >= 0.5)
                        epe = torch.sum((data[view]['flow'] - data[view]['flow_pred']) ** 2, dim=1).sqrt()
                        epe = epe.view(-1)[valid.view(-1)]
                        one_pix = (epe < 1)
                        epe_list.append(epe.mean().item())
                        one_pix_list.append(one_pix.float().mean().item())

        val_epe = np.round(np.mean(np.array(epe_list)), 4)
        val_one_pix = np.round(np.mean(np.array(one_pix_list)), 4)
        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_segmentation_loss = np.round(np.mean(np.array(segmentation_loss_list)), 4)
        logging.info(f"Testing Metrics: epe {val_epe}, 1pix {val_one_pix}, psnr {val_psnr}, seg_loss {val_segmentation_loss}")

    def fetch_data(self):
        try:
            data = next(self.val_iterator)
        except:
            self.val_iterator = iter(self.val_loader)
            data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        self.mask_gen.load_state_dict(ckpt['mask_network'], strict=True)
        logging.info(f"Parameter loading done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/testing.yaml")
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

    bg_categories = ['brick', 'carpet', 'fabric', 'foliage', 'metal', 'tile', 'wallpaper', 'wood']
    base_path = '/data/scratch/ec23984/data-repo/minc/minc-2500/images'
    dataset_dirs = [os.path.join(base_path, label) for label in bg_categories]

    torch.manual_seed(1314)
    np.random.seed(1314)

    tester = Tester(cfg)
    tester.run_test()
