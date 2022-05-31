import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torchgeometry as tgm
from torch import autograd, nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from networks import (GMM, ALIASGenerator, BoundedGridLocNet, GANLoss,
                      MultiscaleDiscriminator, SegGenerator, VGGLoss)
from train_datasets import VITONDataLoader, VITONDataset
from train_options import get_args
from utils import (AverageMeter, cleanup, gen_noise, seed_everything,
                   set_grads, synchronize)


class TrainModel:
    def __init__(self, args):
        self.device = torch.device('cuda', args.local_rank)
        # self.segG = SegGenerator(args, input_nc=args.semantic_nc + 8,
        #                          output_nc=args.semantic_nc).train().to(self.device)
        # self.segD = MultiscaleDiscriminator(args, args.semantic_nc + args.semantic_nc + 8,
        #                                     use_sigmoid=args.no_lsgan).train().to(self.device)
        self.gmm = GMM(args, inputA_nc=7, inputB_nc=3).train().to(self.device)
        self.loc_net = BoundedGridLocNet(args)
        args.update(dict(semantic_nc=7), allow_val_change=True)
        self.aliasG = ALIASGenerator(args, input_nc=3+3+3).train().to(self.device)
        self.aliasD = MultiscaleDiscriminator(args, 3+3+3+3, use_sigmoid=args.no_lsgan,
                                              getIntermFeat=True).train().to(self.device)
        args.update(dict(semantic_nc=13), allow_val_change=True)

        dist.init_process_group(backend="nccl")
        if args.sync_bn:
            # self.segG = nn.SyncBatchNorm.convert_sync_batchnorm(self.segG)
            # self.segD = nn.SyncBatchNorm.convert_sync_batchnorm(self.segD)
            self.gmm = nn.SyncBatchNorm.convert_sync_batchnorm(self.gmm)
            self.aliasG = nn.SyncBatchNorm.convert_sync_batchnorm(self.aliasG)
        # self.segG = DDP(self.segG, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        # self.segD = DDP(self.segD, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        self.gmm = DDP(self.gmm, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        self.aliasG = DDP(self.aliasG, device_ids=[args.local_rank],
                          output_device=args.local_rank, broadcast_buffers=False)

        self.gauss = tgm.image.GaussianBlur((7, 7), (3, 3)).to(self.device)
        self.up = nn.Upsample(size=(args.load_height, args.load_width), mode='bilinear')

        self.criterion_gan = GANLoss(use_lsgan=not args.no_lsgan)
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss().requires_grad_(False).to(self.device)

        # self.optimizer_seg = optim.Adam(list(self.segG.parameters()) +
        #                                 list(self.segD.parameters()), lr=0.0004, betas=(0.5, 0.999))
        # self.optimizer_seg.zero_grad(set_to_none=True)

        self.optimizer_gmm = optim.Adam(self.gmm.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_gmm.zero_grad(set_to_none=True)

        self.opt_aliasG = optim.Adam(self.aliasG.parameters(), lr=0.0001, betas=(0, 0.9))
        self.opt_aliasG.zero_grad(set_to_none=True)

        self.opt_aliasD = optim.Adam(self.aliasD.parameters(), lr=0.0004, betas=(0, 0.9))
        self.opt_aliasD.zero_grad(set_to_none=True)

        self.scaler = amp.GradScaler(enabled=args.use_amp)

        self.parse_labels = {
            0:  ['background',  [0]],
            1:  ['hair',        [1]],
            # contains face and lower body.
            2:  ['paste',       [2, 4, 8, 9, 10, 11, 12]],
            3:  ['upper',       [7]],
            4:  ['noise',       [3]],
            5:  ['left_arm',    [5]],
            6:  ['right_arm',   [6]],
        }

        self.train_dataset = VITONDataset(args)
        self.train_loader = VITONDataLoader(args, self.train_dataset)

        # if args.local_rank==0 and args.use_wandb:
        #     wandb.watch([self.segG, self.segD], log=None)

    def segmentation_train_step(self, args, parse_target_down, parse_agnostic, pose, cloth, cloth_mask, get_img_log=False):
        with amp.autocast(enabled=args.use_amp):
            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(cloth * cloth_mask, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cloth_mask, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down,
                                  gen_noise(cm_down.size(), device=self.device)), dim=1)

            parse_pred_down = self.segG(seg_input)
            parse_target_mx = parse_target_down.argmax(dim=1)

            lambda_ce = 10
            seg_lossG = lambda_ce * self.ce_loss(parse_pred_down, parse_target_mx)

            parse_pred_down = F.softmax(parse_pred_down, dim=1)
            fake_out = self.segD(torch.cat((seg_input, parse_pred_down), dim=1))
            real_out = self.segD(torch.cat((seg_input, parse_target_down.detach()), dim=1))
            # Treat fake images as real to train the Generator.
            seg_lossG += self.criterion_gan(fake_out, True)
            seg_lossD = (self.criterion_gan(real_out, True)     # Treat real as real
                         + self.criterion_gan(fake_out, False))*0.5   # and fake as fake to train Discriminator.

        set_grads(autograd.grad(self.scaler.scale(seg_lossD),
                  self.segD.parameters(), retain_graph=True), self.segD.parameters())
        set_grads(autograd.grad(self.scaler.scale(seg_lossG), self.segG.parameters()), self.segG.parameters())

        self.scaler.step(self.optimizer_seg)
        self.optimizer_seg.zero_grad(set_to_none=True)

        img_log = {}
        if get_img_log:
            parse_pred_mx = parse_pred_down.argmax(dim=1)
            img_log['seg_real'] = (parse_target_mx.detach_()*(255/parse_pred_down.size(1))).cpu().numpy()
            img_log['seg_pred'] = (parse_pred_mx.detach_()*(255/parse_pred_down.size(1))).cpu().numpy()
        return seg_lossG.detach_(), seg_lossD.detach_(), img_log

    def gmm_train_step(self, args, img, img_agnostic, parse, pose, cloth, cloth_mask, get_img_log=False, train=True):
        # Part 2. Clothes Deformation
        with amp.autocast(enabled=args.use_amp):
            cloth_mask_target = parse[:, 3:4]
            cloth_target = ((img+1) * cloth_mask_target) - 1
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='bilinear')
            parse_cloth_gmm = F.interpolate(cloth_mask_target, size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_gmm = F.interpolate(cloth, size=(256, 192), mode='bilinear')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            theta, warped_grid = self.gmm(gmm_input, c_gmm)
            # second-order difference constraint
            warped_c = F.grid_sample(cloth, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')
            if train:
                rx_loss, ry_loss, cx_loss, cy_loss, rg_loss, cg_loss = self.loc_net(theta)
                gmm_loss = self.l1_loss(warped_c, cloth_target) + self.l1_loss(warped_cm, cloth_mask_target) \
                    + torch.mean(0.04*(rx_loss+ry_loss+cx_loss+cy_loss+rg_loss+cg_loss))
            else:
                gmm_loss = torch.zeros(1)

        if train:
            self.scaler.scale(gmm_loss).backward()
            self.scaler.step(self.optimizer_gmm)
            self.optimizer_gmm.zero_grad(set_to_none=True)
        img_log = {}
        if get_img_log:
            img_log['gmm_real'] = (127.5*(cloth_target+1)).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            img_log['gmm_pred'] = (127.5*(warped_c.detach()+1)).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return gmm_loss.detach_(), img_log, warped_c.detach(), warped_cm.detach()

    def alias_train_step(self, args, img, img_agnostic, parse, pose, warped_c, warped_cm, get_img_log=False):
        # Part 3. Try-on synthesis
        with amp.autocast(enabled=args.use_amp):
            misalign_mask = parse[:, 3:4] - warped_cm
            misalign_mask = misalign_mask.clip_(min=0.0)
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 3:4] -= misalign_mask

            alias_inp = torch.cat((img_agnostic, pose, warped_c), dim=1)
            output = self.aliasG(alias_inp, parse, parse_div, misalign_mask)

            fake_out = self.aliasD(torch.cat((alias_inp, output), dim=1))
            real_out = self.aliasD(torch.cat((alias_inp, img.detach()), dim=1))
            # Treat fake images as real to train the Generator.
            alias_lossG = self.criterion_gan(fake_out, True)
            alias_lossD = (self.criterion_gan(real_out, True)     # Treat real as real
                           + self.criterion_gan(fake_out, False))*0.5

            lambda_fm = 10
            lambda_percept = 10

            percept_loss = self.vgg_loss(output, img.detach()) * lambda_percept

            loss_G_GAN_Feat = 0
            for i in range(2):
                for j in range(len(fake_out[i])-1):
                    loss_G_GAN_Feat += self.l1_loss(fake_out[i][j], real_out[i][j].detach()) * (lambda_fm/2)

        set_grads(autograd.grad(self.scaler.scale(alias_lossD),
                  self.aliasD.parameters(), retain_graph=True), self.aliasD.parameters())
        self.scaler.step(self.opt_aliasD)
        self.opt_aliasD.zero_grad(set_to_none=True)

        set_grads(autograd.grad(self.scaler.scale(alias_lossG+loss_G_GAN_Feat+percept_loss),
                  self.aliasG.parameters()), self.aliasG.parameters())
        self.scaler.step(self.opt_aliasG)
        self.opt_aliasG.zero_grad(set_to_none=True)

        img_log = {}
        if get_img_log:
            parse_div_mx = parse_div.argmax(dim=1)
            img_log['parse_div'] = (parse_div_mx.detach_()*(255/parse_div.size(1))).cpu().numpy()
            img_log['img_agnostic'] = (127.5*(img_agnostic+1)).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            img_log['alias_real'] = (127.5*(img+1)).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            img_log['alias_pred'] = (127.5*(output.detach()+1)).type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return alias_lossG.detach_(), alias_lossD.detach_(), percept_loss.detach_(), loss_G_GAN_Feat.detach_(), img_log

    def train_epoch(self, args, epoch):
        self.train_loader.train_sampler.set_epoch(epoch)
        segG_losses = AverageMeter()
        segD_losses = AverageMeter()
        gmm_losses = AverageMeter()
        aliasG_losses = AverageMeter()
        aliasD_losses = AverageMeter()
        percept_losses = AverageMeter()
        fm_losses = AverageMeter()

        img_log = {}
        with tqdm(self.train_loader.data_loader, desc=f"Epoch {epoch:>2}") as pbar:
            for step, batch in enumerate(pbar):
                get_img_log = False
                batch = self.train_loader.device_augment(batch, self.device)
                img = batch['img']
                img_agnostic = batch['img_agnostic']
                parse_target_down_idx = batch['parse_target_down']
                parse_agnostic_idx = batch['parse_agnostic']
                pose = batch['pose']
                cloth = batch['cloth']
                cloth_mask = batch['cloth_mask']
                cloth = ((cloth+1) * cloth_mask) - 1    # mask out the cloth
                parse_target_down = torch.empty(parse_target_down_idx.size(0), args.semantic_nc, 256, 192,
                                                dtype=torch.float, device=self.device).fill_(0.)
                parse_target_down.scatter_(1, parse_target_down_idx.long(), 1.0)
                parse_agnostic = torch.empty(parse_agnostic_idx.size(0), args.semantic_nc, args.load_height, args.load_width,
                                             dtype=torch.float, device=self.device).fill_(0.)
                parse_agnostic.scatter_(1, parse_agnostic_idx.long(), 1.0)
                del parse_target_down_idx, parse_agnostic_idx

                if not step % args.img_log_interval:
                    get_img_log = True

                # seg_lossG, seg_lossD, seg_img_log = self.segmentation_train_step(args, parse_target_down,
                #                                                                  parse_agnostic, pose, cloth, cloth_mask,
                #                                                                  get_img_log=get_img_log)
                # img_log.update(seg_img_log)
                # segG_losses.update(seg_lossG.detach_(), parse_target_down.size(0))
                # segD_losses.update(seg_lossD.detach_(), parse_target_down.size(0))

                # convert 13 channel body parse to 7 channel parse.
                parse_target = self.gauss(self.up(parse_target_down))
                parse_target = parse_target.argmax(dim=1, keepdim=True)
                parse_orig = parse_target.clone()
                for k, v in self.parse_labels.items():
                    for l in v[1]:
                        if l != k:
                            parse_target[parse_orig == l] = k
                del parse_orig

                parse = torch.zeros(parse_target.size(0), 7, args.load_height,
                                    args.load_width, dtype=torch.float32, device=self.device)
                parse.scatter_(1, parse_target, 1.0)

                # with torch.no_grad():
                gmm_loss, gmm_img_log, warped_c, warped_cm = self.gmm_train_step(args, img, img_agnostic, parse,
                                                                                    pose, cloth, cloth_mask,
                                                                                    get_img_log=get_img_log, train=True)
                img_log.update(gmm_img_log)
                gmm_losses.update(gmm_loss.detach_(), cloth.size(0))

                alias_lossG, alias_lossD, percept_loss, loss_G_GAN_Feat, alas_img_log = self.alias_train_step(
                    args, img, img_agnostic, parse, pose, warped_c, warped_cm, get_img_log=get_img_log)

                img_log.update(alas_img_log)
                aliasG_losses.update(alias_lossG)
                aliasD_losses.update(alias_lossD)
                percept_losses.update(percept_loss)
                fm_losses.update(loss_G_GAN_Feat)

                if args.local_rank == 0:
                    if not step % args.log_interval:
                        info = {
                            'SegG Loss': float(segG_losses.avg),
                            'SegD Loss': float(segD_losses.avg),
                            'GMM Loss': float(gmm_losses.avg),
                            'aliasG Loss': float(aliasG_losses.avg),
                            'aliasD Loss': float(aliasD_losses.avg),
                            'percept Loss': float(percept_losses.avg),
                            'fm Loss': float(fm_losses.avg),
                        }
                        pbar.set_postfix(info)
                        if args.use_wandb:
                            wandb.log(info)
                            if get_img_log:
                                im_dict = {}
                                for k in img_log:
                                    im_dict[k] = []
                                    for img in img_log[k]:
                                        im_dict[k].append(wandb.Image(img))
                                wandb.log(im_dict)
                self.scaler.update()
        return info

    def train_loop(self, args):
        for epoch in range(args.init_epoch, args.epochs+1):
            info = self.train_epoch(args, epoch)
            info['epoch'] = epoch
            info['args'] = dict(args)
            if args.local_rank == 0:
                if args.use_wandb:
                    info['run_id'] = wandb.run.id
                    wandb.log({'epoch': epoch})
                self.save_models(args, 'latest', info=info)
                if not epoch % 4:
                    self.save_models(args, epoch, info=info)

    def save_models(self, args, epoch='latest', info={}):
        if args.local_rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            # torch.save(self.segG.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_segG.pth"))
            # torch.save(self.segD.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_segD.pth"))
            torch.save(self.gmm.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_gmm.pth"))
            torch.save(self.aliasG.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_aliasG.pth"))
            torch.save(self.aliasD.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_aliasD.pth"))
            # torch.save(self.optimizer_seg.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_optimizer_seg.pth"))
            torch.save(self.optimizer_gmm.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_optimizer_gmm.pth"))
            torch.save(self.opt_aliasG.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_opt_aliasG.pth"))
            torch.save(self.opt_aliasD.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}_opt_aliasD.pth"))
            if info:
                with open(os.path.join(args.checkpoint_dir, f"{epoch}_info.json"), "w") as f:
                    json.dump(info, f, indent=4)
            print("[+] Weights saved.")

    def load_models(self, args, epoch='latest'):
        synchronize()
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        try:
            # self.segG.load_state_dict(torch.load(os.path.join(
            #     args.checkpoint_dir, "segG.pth"), map_location=map_location))
            # self.segD.load_state_dict(torch.load(os.path.join(
            #     args.checkpoint_dir, "segD.pth"), map_location=map_location))
            self.gmm.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_gmm.pth"), map_location=map_location))
            self.aliasG.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_aliasG.pth"), map_location=map_location))
            self.aliasD.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_aliasD.pth"), map_location=map_location))
            # self.optimizer_seg.load_state_dict(torch.load(os.path.join(
            #     args.checkpoint_dir, f"{epoch}_optimizer_seg.pth"), map_location=map_location))
            self.optimizer_gmm.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_optimizer_gmm.pth"), map_location=map_location))
            self.opt_aliasG.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_opt_aliasG.pth"), map_location=map_location))
            self.opt_aliasD.load_state_dict(torch.load(os.path.join(
                args.checkpoint_dir, f"{epoch}_opt_aliasD.pth"), map_location=map_location))
            if args.local_rank == 0:
                print("[+] Weights loaded.")
        except FileNotFoundError as e:
            if args.local_rank == 0:
                print(f"[!] {e}, skipping weights loading.")


def main():
    args = get_args()
    if args.local_rank == 0:
        print(args)
    torch.cuda.set_device(args.local_rank)
    seed_everything(args.seed)
    try:
        tm = TrainModel(args)
        tm.load_models(args)
        tm.train_loop(args)
        tm.save_models(args)
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Cleaning up, saving and shutting down.")
    finally:
        cleanup()


if __name__ == '__main__':
    main()
