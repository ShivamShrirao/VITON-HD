import os
from os import path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode

np.seterr(divide='ignore', invalid='ignore')


class VITONDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir, args.dataset_mode)
        # load data list
        self.img_names = os.listdir(osp.join(self.data_path, 'image'))

    def __getitem__(self, index):
        img_name = self.img_names[index]
        cloth = cv2.imread(osp.join(self.data_path, 'cloth', img_name), cv2.IMREAD_COLOR)
        cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)
        cloth = cv2.resize(cloth, (self.load_width, self.load_height), interpolation=cv2.INTER_LINEAR)
        cloth = torch.from_numpy(cloth)
        cloth = cloth.permute(2,0,1)
        cloth = (cloth / 127.5).sub_(1)     # [-1,1]

        ext = img_name.split('.')[-1]
        cmask_name = img_name.replace(f'.{ext}', '.png')

        cloth_mask = cv2.imread(osp.join(self.data_path, 'cloth-mask', cmask_name), cv2.IMREAD_GRAYSCALE)
        cloth_mask = cv2.resize(cloth_mask, (self.load_width, self.load_height), interpolation=cv2.INTER_NEAREST)
        cloth_mask = torch.from_numpy(cloth_mask)
        cloth_mask = (cloth_mask / 255.)    # [0,1]
        cloth_mask.unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace(f'.{ext}', '_rendered.png')
        pose_rgb = cv2.imread(osp.join(self.data_path, 'openpose-img', pose_name), cv2.IMREAD_COLOR)
        pose_rgb = cv2.cvtColor(pose_rgb, cv2.COLOR_BGR2RGB)
        pose_rgb = cv2.resize(pose_rgb, (self.load_width, self.load_height), interpolation=cv2.INTER_LINEAR)
        pose_rgb = torch.from_numpy(pose_rgb)
        pose_rgb = pose_rgb.permute(2,0,1)
        pose_rgb = (pose_rgb / 127.5).sub_(1)   # [-1,1]

        # load parsing image
        parse_name = img_name.replace(f'.{ext}', '.png')
        parse_down = cv2.imread(osp.join(self.data_path, 'parse-down', parse_name), cv2.IMREAD_UNCHANGED)
        parse_down = torch.from_numpy(parse_down[None]).to(torch.uint8)

        parse_agnostic = cv2.imread(osp.join(self.data_path, 'parse-agnostic', parse_name), cv2.IMREAD_UNCHANGED)
        parse_agnostic = torch.from_numpy(parse_agnostic[None]).to(torch.uint8)

        # load person image
        img = cv2.imread(osp.join(self.data_path, 'image', img_name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.load_width, self.load_height), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img = (img / 127.5).sub_(1)     # [-1,1]

        img_agnostic = cv2.imread(osp.join(self.data_path, 'image-agnostic', img_name), cv2.IMREAD_COLOR)
        img_agnostic = cv2.cvtColor(img_agnostic, cv2.COLOR_BGR2RGB)
        img_agnostic = cv2.resize(img_agnostic, (self.load_width, self.load_height), interpolation=cv2.INTER_LINEAR)
        img_agnostic = torch.from_numpy(img_agnostic)
        img_agnostic = img_agnostic.permute(2,0,1)
        img_agnostic = (img_agnostic / 127.5).sub_(1)   # [-1,1]

        result = {
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_target_down': parse_down,
            'parse_agnostic': parse_agnostic,
            'pose': pose_rgb,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
        }
        return result

    def __len__(self):
        return len(self.img_names)


class VITONDataLoader:
    def __init__(self, args, dataset):
        super().__init__()

        self.train_sampler = data.distributed.DistributedSampler(dataset, shuffle=args.shuffle)

        self.data_loader = data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=False,
            sampler=self.train_sampler, collate_fn=lambda x: x
        )
        self.dataset = dataset

    def device_augment(self, data_batch, device, angle1_keys=[]): #'cloth', 'cloth_mask']):
        batch = {}
        flip = torch.randn([]) > 0.5
        rot = torch.randn([]) > 0.5
        angle1, angle2 = torch.randint(-15, 15, (2,))
        angle = 0
        for key in data_batch[0].keys():
            batch[key] = torch.stack([inpd[key].to(device, non_blocking=True)
                                     for inpd in data_batch])
            if flip:
                batch[key] = TF.hflip(batch[key])
            if rot:
                angle = angle1 if key in angle1_keys else angle2
            batch[key] = TF.rotate(batch[key], float(angle))
        return batch
