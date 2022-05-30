import os
import json
import argparse
import wandb

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100, metavar="N", help="Number of epochs.")
    parser.add_argument('--load_height', type=int, default=1024, help="Height of input image.")
    parser.add_argument('--load_width', type=int, default=768, help="Width of input image.")
    parser.add_argument('--sync_bn', type=bool, default=False, help="Synchronize BatchNorm across all devices. Use when batchsize is small.")
    parser.add_argument('--use_amp', type=bool, default=True, help="Use mixed precision training.")

    parser.add_argument('--use_wandb', type=bool, default=False, help="Use wandb logger.")
    parser.add_argument('--project', type=str, default='VITON-HD', help="Name of wandb project.")
    parser.add_argument('--resume', type=str, default=None, help="ID of wandb run to resume.")
    parser.add_argument('--init_epoch', type=int, default=1, metavar="N", help="Initial epoch.")
    parser.add_argument('--log_interval', type=int, default=20, metavar="N", help="Log per N steps.")
    parser.add_argument('--img_log_interval', type=int, default=400, metavar="N", help="Log image per N steps.")
    parser.add_argument('--seed', type=int, default=3407, metavar="N", help="Random seed.")
    parser.add_argument('--shuffle', type=bool, default=True, help="Shuffle the dataset.")

    parser.add_argument('--workers', type=int, default=4, metavar="N", help="Number of workers for dataloader.")
    try:
        dataset_dir = os.environ['SM_CHANNEL_TRAIN']
    except KeyError:
        dataset_dir = 'datasets/'
    parser.add_argument('--dataset_dir', type=str, default=dataset_dir)
    parser.add_argument('--dataset_mode', type=str, default='train', help="train or test.")
    parser.add_argument('--checkpoint_dir', type=str, default="/opt/ml/checkpoints/")

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--no_lsgan', type=bool, default=False, help="Not use Least Squares GAN loss.")

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5, help="For GMM.")

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    args = parser.parse_args()
    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        args.local_rank = 0
    info_pth = os.path.join(args.checkpoint_dir, f"latest_info.json")
    if os.path.exists(info_pth):
        with open(info_pth, 'r') as fp:
            info = json.load(fp)
        updict = dict(init_epoch=info['epoch'] + 1, local_rank=args.local_rank)
        vars(args).update(info['args'])
        vars(args).update(updict)
        if args.use_wandb:
            if args.local_rank == 0:
                wandb.init(id=info['run_id'], project=args.project, resume='must')
                wandb.config.update(updict, allow_val_change=True)
                args = wandb.config
                
    elif args.use_wandb:
        run = wandb.init(project=args.project, config=args)
        args = wandb.config

    return args