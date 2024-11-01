import argparse
import os

import torch

from yowof.config import build_dataset_config, build_model_config
from yowof.dataset.transforms import BaseTransform
from yowof.evaluator.ava_evaluator import AVA_Evaluator
from yowof.evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from yowof.evaluator.custom_evaluator import Custom_Evaluator
from yowof.models.detector import build_model
from yowof.utils.misc import CollateFunc, load_weight


def parse_args():
    parser = argparse.ArgumentParser(description="YOWOF")

    # basic
    parser.add_argument(
        "-size", "--img_size", default=320, type=int, help="the size of input frame"
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda.")
    parser.add_argument(
        "-mt",
        "--metrics",
        default=["frame_map", "video_map"],
        type=str,
        help="evaluation metrics",
    )
    parser.add_argument(
        "--save_path",
        default="results/",
        type=str,
        help="Trained state_dict file path to open",
    )

    # model
    parser.add_argument(
        "-v", "--version", default="yowof-r18", type=str, help="build yowof"
    )
    parser.add_argument(
        "--weight", default=None, type=str, help="Trained state_dict file path to open"
    )
    parser.add_argument("--topk", default=50, type=int, help="NMS threshold")

    # dataset
    parser.add_argument(
        "-d", "--dataset", default="ucf24", help="ucf24, jhmdb21, ava_v2.2."
    )

    # eval
    parser.add_argument(
        "--cal_frame_mAP",
        action="store_true",
        default=False,
        help="calculate frame mAP.",
    )
    parser.add_argument(
        "--cal_video_mAP",
        action="store_true",
        default=False,
        help="calculate video mAP.",
    )

    return parser.parse_args()


def ucf_jhmdb_eval(device, args, d_cfg, model, transform):
    evaluator = UCF_JHMDB_Evaluator(
        device=device,
        dataset=args.dataset,
        model_name=args.version,
        data_root=d_cfg["data_root"],
        img_size=d_cfg["test_size"],
        len_clip=d_cfg["len_clip"],
        conf_thresh=0.01,
        iou_thresh=0.5,
        transform=transform,
        gt_folder=d_cfg["gt_folder"],
        save_path=args.save_path,
    )

    if args.cal_frame_mAP:
        evaluator.evaluate_frame_map(model, show_pr_curve=True)
    if args.cal_video_mAP:
        evaluator.evaluate_video_map(model)


def ava_eval(device, d_cfg, model, transform, version="v2.2"):
    evaluator = AVA_Evaluator(
        device=device,
        d_cfg=d_cfg,
        img_size=d_cfg["test_size"],
        len_clip=d_cfg["len_clip"],
        sampling_rate=d_cfg["sampling_rate"],
        transform=transform,
        collate_fn=CollateFunc(),
        full_test_on_val=False,
        version=version,
    )

    evaluator.evaluate_frame_map(model)


def custom_eval(device, args, d_cfg, model, transform):
    evaluator = Custom_Evaluator(
        device=device,
        dataset=args.dataset,
        model_name=args.version,
        data_root=d_cfg["data_root"],
        img_size=d_cfg["test_size"],
        len_clip=d_cfg["len_clip"],
        conf_thresh=0.01,
        iou_thresh=0.5,
        transform=transform,
    )

    evaluator.evaluate_frame_map(model)


if __name__ == "__main__":
    args = parse_args()
    # dataset
    if args.dataset == "ucf24":
        num_classes = 24

    elif args.dataset == "jhmdb":
        num_classes = 21

    elif args.dataset == "ava_v2.2":
        num_classes = 80
        version = "v2.2"

    elif args.dataset == "ava_pose":
        num_classes = 14
        version = "pose"

    elif args.dataset == "custom":
        num_classes = 2

    else:
        print("unknow dataset.")
        exit(0)

    # cuda
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False,
        eval_mode=True,
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg["test_size"],
        pixel_mean=d_cfg["pixel_mean"],
        pixel_std=d_cfg["pixel_std"],
    )

    # run
    if args.dataset in ["ucf24", "jhmdb21"]:
        ucf_jhmdb_eval(
            device=device, args=args, d_cfg=d_cfg, model=model, transform=basetransform
        )
    elif args.dataset in ["ava_v2.2", "ava_pose"]:
        ava_eval(
            device=device,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            version=version,
        )
    elif args.dataset == "custom":
        custom_eval(
            device=device, args=args, d_cfg=d_cfg, model=model, transform=basetransform
        )
