from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render
from lib.mask_generator import UNet

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time


class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.model.cuda()

        # Mask predictor: input concattenated over channel, 3-channel output
        # Mask predictor: input must be concattenated over channel, 3-channel output
        self.mask_gen = UNet(n_channels=3, n_classes=3)
        self.mask_gen.cuda()

        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase, mask_network=self.mask_gen)
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
        self.mask_gen.eval()

    def infer_seqence(self, view_select, ratio=0.5):
            total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
            for idx in tqdm(range(1)):
                start_time = time.perf_counter()
                item = self.dataset.get_test_item(idx, source_id=view_select, require_mask=True, predict_mask=True, hr_img=True)
                data = self.fetch_data(item)
                start_time = time.perf_counter()
                data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
                with torch.no_grad():
                    data, _, _ = self.model(data, is_train=False)
                    data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = self.tensor2np(data['novel_view']['img_pred'])
                # End time measurement
                end_time = time.perf_counter()
                cv2.imwrite(self.cfg.test_out_path + f"/{data['name']}-f{idx}_novel.jpg", render_novel)

                # Calculate elapsed 
                elapsed_time = end_time - start_time

                # Print elapsed 
                print(f'Frame processing time: {elapsed_time:.6f} seconds')

                # Novel mask generation
                # Concatenate the novel view prediction with itself
                img_pred_concat = torch.cat([data['novel_view']['img_pred'], data['novel_view']['img_pred']], dim=0)

                # Pass the concatenated tensor through the mask generator
                with torch.no_grad():
                    masks = self.mask_gen(img_pred_concat)

                # Since both masks are for the same image, take the first one to sample
                tmp_mask = masks[1].detach().sigmoid()

                # Thresholded mask
                threshold = 0.5
                tmp_mask_thresholded = (tmp_mask > threshold).float() * 255
                tmp_mask_thresholded = tmp_mask_thresholded.permute(1, 2, 0).cpu().numpy()
                tmp_mask_thresholded.astype(np.uint8)

                # Save the mask to the same directory as the novel image
                cv2.imwrite(self.cfg.test_out_path + f"/{data['name']}-f{idx}_mask.jpg", tmp_mask_thresholded)

             
            
    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    cfg.test_out_path = './test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    render.infer_seqence(view_select=arg.src_view, ratio=arg.ratio)
