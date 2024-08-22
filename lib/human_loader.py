from torch.utils.data import Dataset

import numpy as np
import os
from PIL import Image
import cv2
import torch
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from pathlib import Path
import logging
import json
from tqdm import tqdm


def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm


def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                          torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    offset0 = new_intr1[0, 2] - new_intr0[0, 2]
    disparity0 = -new_depth0 * Tf_x
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15


class StereoHumanDataset(Dataset):
    def __init__(self, opt, phase='train', mask_network=None):
        self.opt = opt
        self.use_processed_data = opt.use_processed_data
        self.phase = phase
        if self.phase == 'train':
            self.data_root = os.path.join(opt.data_root, 'train')
        elif self.phase == 'val':
            self.data_root = os.path.join(opt.data_root, 'val')
        elif self.phase == 'test':
            self.data_root = opt.test_data_root

        self.mask_network = mask_network
        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.img_hr_path = os.path.join(self.data_root, 'img/%s/%d_hr.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%d_extrinsic.npy')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))

        if self.use_processed_data:
            self.local_data_root = os.path.join(opt.data_root, 'rectified_local', self.phase)
            self.local_img_path = os.path.join(self.local_data_root, 'img/%s/%d.jpg')
            self.local_mask_path = os.path.join(self.local_data_root, 'mask/%s/%d.png')
            self.local_flow_path = os.path.join(self.local_data_root, 'flow/%s/%d.npy')
            self.local_valid_path = os.path.join(self.local_data_root, 'valid/%s/%d.png')
            self.local_parm_path = os.path.join(self.local_data_root, 'parm/%s/%d_%d.json')

            # if os.path.exists(self.local_data_root):
            #     assert len(os.listdir(os.path.join(self.local_data_root, 'img'))) == len(self.sample_list)
            #     logging.info(f"Using local data in {self.local_data_root} ...")
            # else:
            #     self.save_local_stereo_data()

    def save_local_stereo_data(self):
        logging.info(f"Generating data to {self.local_data_root} ...")
        for sample_name in tqdm(self.sample_list):
            view0_data = self.load_single_view(sample_name, self.opt.source_id[0], hr_img=False,
                                               require_mask=True, require_pts=True)
            view1_data = self.load_single_view(sample_name, self.opt.source_id[1], hr_img=False,
                                               require_mask=True, require_pts=True)
            lmain_stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)

            for sub_dir in ['/img/', '/mask/', '/flow/', '/valid/', '/parm/']:
                Path(self.local_data_root + sub_dir + str(sample_name)).mkdir(exist_ok=True, parents=True)

            img0_save_name = self.local_img_path % (sample_name, self.opt.source_id[0])
            mask0_save_name = self.local_mask_path % (sample_name, self.opt.source_id[0])
            img1_save_name = self.local_img_path % (sample_name, self.opt.source_id[1])
            mask1_save_name = self.local_mask_path % (sample_name, self.opt.source_id[1])
            flow0_save_name = self.local_flow_path % (sample_name, self.opt.source_id[0])
            valid0_save_name = self.local_valid_path % (sample_name, self.opt.source_id[0])
            flow1_save_name = self.local_flow_path % (sample_name, self.opt.source_id[1])
            valid1_save_name = self.local_valid_path % (sample_name, self.opt.source_id[1])
            parm_save_name = self.local_parm_path % (sample_name, self.opt.source_id[0], self.opt.source_id[1])

            Image.fromarray(lmain_stereo_np['img0']).save(img0_save_name, quality=95)
            Image.fromarray(lmain_stereo_np['mask0']).save(mask0_save_name)
            Image.fromarray(lmain_stereo_np['img1']).save(img1_save_name, quality=95)
            Image.fromarray(lmain_stereo_np['mask1']).save(mask1_save_name)
            np.save(flow0_save_name, lmain_stereo_np['flow0'].astype(np.float16))
            Image.fromarray(lmain_stereo_np['valid0']).save(valid0_save_name)
            np.save(flow1_save_name, lmain_stereo_np['flow1'].astype(np.float16))
            Image.fromarray(lmain_stereo_np['valid1']).save(valid1_save_name)
            save_np_to_json(lmain_stereo_np['camera'], parm_save_name)

        logging.info("Generating data Done!")

    def load_local_stereo_data(self, sample_name):
        img0_name = self.local_img_path % (sample_name, self.opt.source_id[0])
        mask0_name = self.local_mask_path % (sample_name, self.opt.source_id[0])
        img1_name = self.local_img_path % (sample_name, self.opt.source_id[1])
        mask1_name = self.local_mask_path % (sample_name, self.opt.source_id[1])
        flow0_name = self.local_flow_path % (sample_name, self.opt.source_id[0])
        flow1_name = self.local_flow_path % (sample_name, self.opt.source_id[1])
        valid0_name = self.local_valid_path % (sample_name, self.opt.source_id[0])
        valid1_name = self.local_valid_path % (sample_name, self.opt.source_id[1])
        parm_name = self.local_parm_path % (sample_name, self.opt.source_id[0], self.opt.source_id[1])

        stereo_data = {
            'img0': read_img(img0_name),
            'mask0': read_img(mask0_name),
            'img1': read_img(img1_name),
            'mask1': read_img(mask1_name),
            'camera': load_json_to_np(parm_name),
            'flow0': np.load(flow0_name),
            'valid0': read_img(valid0_name),
            'flow1': np.load(flow1_name),
            'valid1': read_img(valid1_name)
        }

        return stereo_data

    def load_single_view(self, sample_name, source_id, hr_img=False, require_mask=True, require_pts=True):
        img_name = self.img_path % (sample_name, source_id)
        image_hr_name = self.img_hr_path % (sample_name, source_id)
        mask_name = self.mask_path % (sample_name, source_id)
        depth_name = self.depth_path % (sample_name, source_id)
        intr_name = self.intr_path % (sample_name, source_id)
        extr_name = self.extr_path % (sample_name, source_id)

        intr, extr = np.load(intr_name), np.load(extr_name)
        mask, pts = None, None
        if hr_img:
            img = read_img(image_hr_name)
            intr[:2] *= 2
        else:
            img = read_img(img_name)
        if require_mask:
            mask = read_img(mask_name)
        if require_pts and os.path.exists(depth_name):
            depth = read_depth(depth_name)
            pts = depth2pts(torch.FloatTensor(depth), torch.FloatTensor(extr), torch.FloatTensor(intr))

        return img, mask, intr, extr, pts

    def get_novel_view_tensor(self, sample_name, view_id):
        img, _, intr, extr, _ = self.load_single_view(sample_name, view_id, hr_img=self.opt.use_hr_img,
                                                      require_mask=False, require_pts=False)
        width, height = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view_id': torch.IntTensor([view_id]),
            'img': img,
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }

        return novel_view_data

    def get_rectified_stereo_data(self, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0, pts0 = main_view_data
        img1, mask1, intr1, extr1, pts1 = ref_view_data

        H, W = 1024, 1024
        r0, t0 = extr0[:3, :3], extr0[:3, 3:]
        r1, t1 = extr1[:3, :3], extr1[:3, 3:]
        inv_r0 = r0.T
        inv_t0 = - r0.T @ t0
        E0 = np.eye(4)
        E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
        E1 = np.eye(4)
        E1[:3, :3], E1[:3, 3:] = r1, t1
        E = E1 @ E0
        R, T = E[:3, :3], E[:3, 3]
        dist0, dist1 = np.zeros(4), np.zeros(4)

        R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)

        new_extr0 = R0 @ extr0
        new_intr0 = P0[:3, :3]
        new_extr1 = R1 @ extr1
        new_intr1 = P1[:3, :3]
        Tf_x = np.array(P1[0, 3])

        camera = {
            'intr0': new_intr0,
            'intr1': new_intr1,
            'extr0': new_extr0,
            'extr1': new_extr1,
            'Tf_x': Tf_x
        }

        rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
        new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)
        new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        rectify0 = new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y
        rectify1 = new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y

        stereo_data = {
            'img0': new_img0,
            'mask0': new_mask0,
            'img1': new_img1,
            'mask1': new_mask1,
            'camera': camera
        }

        if pts0 is not None:
            flow0, flow1 = stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x)

            kernel = np.ones((3, 3), dtype=np.uint8)
            flow_eroded, valid_eroded = [], []
            for (flow, new_mask) in [(flow0, new_mask0), (flow1, new_mask1)]:
                valid = (new_mask.copy()[:, :, 0] / 255.0).astype(np.float32)
                valid = cv2.erode(valid, kernel, 1)
                valid[valid >= 0.66] = 1.0
                valid[valid < 0.66] = 0.0
                flow *= valid
                valid *= 255.0
                flow_eroded.append(flow)
                valid_eroded.append(valid)

            stereo_data.update({
                'flow0': flow_eroded[0],
                'valid0': valid_eroded[0].astype(np.uint8),
                'flow1': flow_eroded[1],
                'valid1': valid_eroded[1].astype(np.uint8)
            })

        return stereo_data

    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'Tf_x': torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'Tf_x': -torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        if 'flow0' in stereo_data:
            flow_tensor, valid_tensor = [], []
            for (flow_view, valid_view) in [('flow0', 'valid0'), ('flow1', 'valid1')]:
                flow = torch.from_numpy(stereo_data[flow_view])
                flow = torch.unsqueeze(flow, dim=0)
                flow_tensor.append(flow)

                valid = torch.from_numpy(stereo_data[valid_view])
                valid = torch.unsqueeze(valid, dim=0)
                valid = valid / 255.0
                valid_tensor.append(valid)

            lmain_data['flow'], lmain_data['valid'] = flow_tensor[0], valid_tensor[0]
            rmain_data['flow'], rmain_data['valid'] = flow_tensor[1], valid_tensor[1]

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}

    def get_item(self, index, novel_id=None):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if self.use_processed_data:
            stereo_np = self.load_local_stereo_data(sample_name)
        else:
            view0_data = self.load_single_view(sample_name, self.opt.source_id[0], hr_img=False,
                                               require_mask=True, require_pts=True)
            view1_data = self.load_single_view(sample_name, self.opt.source_id[1], hr_img=False,
                                               require_mask=True, require_pts=True)
            stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        if novel_id:
            novel_id = np.random.choice(novel_id)
            dict_tensor.update({
                'novel_view': self.get_novel_view_tensor(sample_name, novel_id)
            })

        return dict_tensor

    def get_test_item(self, index, source_id, require_mask=True, predict_mask=False, hr_img=False):
        sample_id = index % len(self.sample_list)
        sample_name = self.sample_list[sample_id]

        if self.use_processed_data:
            logging.error('test data loader not support processed data')

        view0_data = self.load_single_view(sample_name, source_id[0], hr_img=False, require_mask=require_mask, require_pts=False)
        view1_data = self.load_single_view(sample_name, source_id[1], hr_img=False, require_mask=require_mask, require_pts=False)

        if predict_mask:
            if not self.mask_network:
                print('Mask network is not defined.')
            else:
                view0_mask, view1_mask = self._get_masks(view0_data, view1_data)
                view0_data = (view0_data[0], view0_mask, view0_data[2], view0_data[3], view0_data[4])
                view1_data = (view1_data[0], view1_mask, view1_data[2], view1_data[3], view1_data[4])

        lmain_intr_ori, lmain_extr_ori = view0_data[2], view0_data[3]
        rmain_intr_ori, rmain_extr_ori = view1_data[2], view1_data[3]
        stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, sample_name)

        dict_tensor['lmain']['intr_ori'] = torch.FloatTensor(lmain_intr_ori)
        dict_tensor['rmain']['intr_ori'] = torch.FloatTensor(rmain_intr_ori)
        dict_tensor['lmain']['extr_ori'] = torch.FloatTensor(lmain_extr_ori)
        dict_tensor['rmain']['extr_ori'] = torch.FloatTensor(rmain_extr_ori)

        image_hr_name = self.img_hr_path % (sample_name, self.opt.val_novel_id[0])

        img_len = 2048 if self.opt.use_hr_img else 1024
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len]),
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })
       
        return dict_tensor
    
    
    def _get_masks(self, view0_data, view1_data):
        # Normalize the data from [0, 255] to [-1, 1]
        print(f'unprocessed img {view0_data[0].min()}, {view0_data[0].max()}')
        view0_data0 = 2 * (view0_data[0] / 255.0) - 1
        view0_data1 = 2 * (view1_data[0] / 255.0) - 1
        print(f'norm img {view0_data0.min()}, {view0_data0.max()}')

        # Transpose the data from (H, W, C) to (C, H, W) and concatenate along the batch dimension
        view0_data0 = np.transpose(view0_data0, (2, 0, 1))
        view0_data1 = np.transpose(view0_data1, (2, 0, 1))

        # Convert to tensor and concatenate along the batch dimension
        tensor0 = torch.tensor(view0_data0, dtype=torch.float32).unsqueeze(0).cuda()
        tensor1 = torch.tensor(view0_data1, dtype=torch.float32).unsqueeze(0).cuda()

        concatenated_tensor = torch.cat([tensor0, tensor1], dim=0)
        print('model input shape: ', concatenated_tensor.shape)

        # Forward pass through the mask network
        logits = self.mask_network(concatenated_tensor)
        print(f'model output shape {logits.shape}')

        # Convert the logits back to numpy arrays and transpose to (H, W, C)
        logits_left = logits[0].squeeze(0).cpu().detach().sigmoid().numpy().transpose(1, 2, 0) 
        logits_right = logits[1].squeeze(0).cpu().detach().sigmoid().numpy().transpose(1, 2, 0)
        print(f'mask left shape {logits_left.shape}')
        print(f'{logits_left.min()}, {logits_left.max()}')


        threshold = 0.5
        mask_left = (logits_left > (threshold)).astype(np.uint8) * 255
        mask_right = (logits_right > (threshold)).astype(np.uint8) * 255

        # Save one of the masks to the folder "test_out"
        os.makedirs('test_out', exist_ok=True)
        mask_path = os.path.join('test_out', 'mask_left.jpg')
        cv2.imwrite(mask_path, mask_left)
        mask_path = os.path.join('test_out', 'mask_right.jpg')
        cv2.imwrite(mask_path, mask_right)

        return mask_left, mask_right


    def __getitem__(self, index):
        if self.phase == 'train':
            return self.get_item(index, novel_id=self.opt.train_novel_id)
        elif self.phase == 'val' or self.phase == 'test':
            return self.get_item(index, novel_id=self.opt.val_novel_id)

    def __len__(self):
        self.train_boost = 50
        self.val_boost = 100
        if self.phase == 'train':
            return len(self.sample_list) * self.train_boost
        elif self.phase == 'val':
            return len(self.sample_list) * self.val_boost
        else:
            return len(self.sample_list)
