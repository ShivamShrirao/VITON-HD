import os
import argparse
import wandb

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100, metavar="N", help="Number of epochs.")
    parser.add_argument('--load_height', type=int, default=1024, help="Height of input image.")
    parser.add_argument('--load_width', type=int, default=768, help="Width of input image.")
    parser.add_argument('--sync_bn', action='store_true', help="Synchronize BatchNorm across all devices. Use when batchsize is small.")
    parser.add_argument('--use_amp', action='store_true', help="Use mixed precision training.")

    parser.add_argument('--use_wandb', action='store_true', help="Use wandb logger.")
    parser.add_argument('--project', type=str, default='VITON-HD', help="Name of wandb project.")
    parser.add_argument('--resume', type=str, default=None, help="ID of wandb run to resume.")
    parser.add_argument('--init_epoch', type=int, default=1, metavar="N", help="Initial epoch.")
    parser.add_argument('--log_interval', type=int, default=20, metavar="N", help="Log per N steps.")
    parser.add_argument('--seed', type=int, default=3407, metavar="N", help="Random seed.")
    parser.add_argument('--shuffle', action='store_true', help="Shuffle the dataset.")

    parser.add_argument('--workers', type=int, default=4, metavar="N", help="Number of workers for dataloader.")
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='train', help="train or test.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')

    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--no_lsgan', action='store_true', help="Not use Least Squares GAN loss.")

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
    if args.use_wandb and args.local_rank == 0:
        if args.resume:
            run = wandb.init(id=args.resume, project=args.project, resume='must')
            wandb.config.update(dict(init_epoch=run.summary['epoch']+1, resume=args.resume), allow_val_change=True)
        else:
            run = wandb.init(project=args.project)
            wandb.config.update(args)
        args = wandb.config
    return args