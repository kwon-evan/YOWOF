# Train YOWOF-R18
python train.py \
        --cuda \
        -d jhmdb \
        -v yowof-r18 \
        --num_workers 4 \
        --eval_epoch 2 \
        --eval \
