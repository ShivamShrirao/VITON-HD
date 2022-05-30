torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        train.py \
        --use_amp \
        --epochs 100 \
        --batch_size 24 \
        --project VITON-HD \
        --shuffle \
        --load_height 1024 \
        --load_width 768 \
        --use_wandb \
        --log_interval 20 \
        --workers 8 \
        --dataset_dir /home/azureuser/data/docker_data
