# Train YOWO-r50
python -m torch.distributed.run --nproc_per_node=4 train.py \
                                                  -dist \
                                                  --cuda \
                                                  -d custom \
                                                  -v yowof-r50 \
                                                  --num_workers 8 \
                                                  --eval_epoch 1 \
                                                  --eval \
                                                  --fp16 \
