# Train YOWO-D19
python train.py \
        --cuda \
        -d ava_v2.2 \
        -v yowof-r18 \
        --num_workers 4 \
        --eval_epoch 1 \
        --eval \
        # --fp16 \
