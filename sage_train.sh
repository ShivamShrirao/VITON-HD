pip3 install -r code/requirements.txt

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
        train.py \
        "$@"
