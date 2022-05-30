torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        train.py \
        --epochs 100 \
        --batch_size 2 \
        --project VITON-HD \
        --load_height 1024 \
        --load_width 768 \
        --use_wandb True \
        --workers 8 \
        --dataset_dir /home/ubuntu/data/datadrive/docker_data/ \
        --checkpoint_dir checkpoints/
