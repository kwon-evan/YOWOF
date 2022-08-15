python eval.py \
        --cuda \
        -d ucf24 \
        -v yowof-r18 \
        -size 320 \
        --gt_folder ./evaluator/groundtruths_ucf_jhmdb/groundtruths_ucf/ \
        --dt_folder ./results/ucf_detections/yowof-r18/detections_1/ \
        --save_path ./evaluator/eval_results/ \
        --weight ./weights/ucf24/yowof-r18/yowof-r18_epoch_1_92.5_94.8.pth \
        --cal_mAP \
        # --redo \
