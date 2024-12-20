import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from yowof.config import build_dataset_config, build_model_config
from yowof.dataset.transforms import BaseTransform
from yowof.models.detector import build_model
from yowof.utils.misc import load_weight


def parse_args():
    parser = argparse.ArgumentParser(description="YOWOF")

    # basic
    parser.add_argument(
        "-size", "--img_size", default=320, type=int, help="the size of input frame"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="show the visulization results.",
    )
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda.")
    parser.add_argument(
        "--save_folder", default="det_results/", type=str, help="Dir to save results"
    )
    parser.add_argument(
        "-vs",
        "--vis_thresh",
        default=0.35,
        type=float,
        help="threshold for visualization",
    )
    parser.add_argument(
        "--video", default="9Y_l9NsnYE0.mp4", type=str, help="AVA video name."
    )
    parser.add_argument("-d", "--dataset", default="ava_v2.2", help="ava_v2.2")

    # model
    parser.add_argument(
        "-v", "--version", default="yowof-r18", type=str, help="build yowof"
    )
    parser.add_argument(
        "--weight", default=None, type=str, help="Trained state_dict file path to open"
    )
    parser.add_argument("--topk", default=40, type=int, help="NMS threshold")
    parser.add_argument(
        "-inf",
        "--inf_mode",
        default="clip",
        type=str,
        choices=["clip", "semi_stream", "stream"],
        help="inference mode: clip or stream",
    )

    return parser.parse_args()


@torch.no_grad()
def detect_clip(args, d_cfg, model, device, transform, class_names):
    # path to save
    save_path = os.path.join(args.save_folder, "ava_video")
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(d_cfg["data_root"], "videos_15min", args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    save_size = (640, 480)
    save_name = os.path.join(save_path, "detection.avi")
    fps = 15.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # run
    video_clip = []
    while True:
        ret, frame = video.read()

        if ret:
            # to PIL image
            frame_pil = Image.fromarray(frame.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg["len_clip"]):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [T, 3, H, W]
            x = torch.stack(x)
            x = x.unsqueeze(0).to(device)  # [B, T, 3, H, W], B=1

            t0 = time.time()
            # inference
            out_bboxes = model(x)
            out_bboxes = out_bboxes[0]
            print("inference time ", time.time() - t0, "s")

            # visualize detection results
            for bbox in out_bboxes:
                x1, y1, x2, y2 = bbox[:4]
                cls_out = bbox[4:]

                # rescale bbox
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

                cls_scores = np.array(cls_out)
                indices = np.where(cls_scores > args.vis_thresh)
                scores = cls_scores[indices]
                indices = list(indices[0])
                scores = list(scores)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if len(scores) > 0:
                    blk = np.zeros(frame.shape, np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    coord = []
                    text = []
                    text_size = []

                    for _, cls_ind in enumerate(indices):
                        text.append(
                            "[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind])
                        )
                        text_size.append(
                            cv2.getTextSize(
                                text[-1], font, fontScale=0.25, thickness=1
                            )[0]
                        )
                        coord.append((x1 + 3, y1 + 7 + 10 * _))
                        cv2.rectangle(
                            blk,
                            (coord[-1][0] - 1, coord[-1][1] - 6),
                            (
                                coord[-1][0] + text_size[-1][0] + 1,
                                coord[-1][1] + text_size[-1][1] - 4,
                            ),
                            (0, 255, 0),
                            cv2.FILLED,
                        )
                    frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                    for t in range(len(text)):
                        cv2.putText(frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)

            # save
            out.write(frame)

            if args.show:
                # show
                cv2.imshow("key-frame detection", frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


@torch.no_grad()
def detect_stream(args, d_cfg, model, device, transform, class_names):
    # path to save
    save_path = os.path.join(args.save_folder, "ava_video")
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(d_cfg["data_root"], "videos_15min", args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    save_size = (640, 480)
    save_name = os.path.join(save_path, "detection.avi")
    fps = 15.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # initalize model
    model.initialization = True

    # run
    video_clip = []
    while True:
        ret, frame = video.read()

        if ret:
            # to PIL image
            frame_pil = Image.fromarray(frame.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg["len_clip"]):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [T, 3, H, W]
            x = torch.stack(x)
            x = x.unsqueeze(0).to(device)  # [B, T, 3, H, W], B=1

            t0 = time.time()
            # inference
            out_bboxes = model(x)
            print("inference time ", time.time() - t0, "s")

            # visualize detection results
            for bbox in out_bboxes:
                x1, y1, x2, y2 = bbox[:4]
                cls_out = bbox[4:]

                # rescale bbox
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

                cls_scores = np.array(cls_out)
                indices = np.where(cls_scores > args.vis_thresh)
                scores = cls_scores[indices]
                indices = list(indices[0])
                scores = list(scores)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if len(scores) > 0:
                    blk = np.zeros(key_frame.shape, np.uint8)
                    font = cv2.LINE_AA
                    coord = []
                    text = []
                    text_size = []
                    # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
                    for _, cls_ind in enumerate(indices):
                        text.append(
                            "[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind])
                        )
                        text_size.append(
                            cv2.getTextSize(text[-1], font, fontScale=0.5, thickness=1)[
                                0
                            ]
                        )
                        coord.append((x1 + 3, y1 + 14 + 20 * _))
                        cv2.rectangle(
                            blk,
                            (coord[-1][0] - 1, coord[-1][1] - 12),
                            (
                                coord[-1][0] + text_size[-1][0] + 1,
                                coord[-1][1] + text_size[-1][1] - 4,
                            ),
                            (0, 255, 0),
                            cv2.FILLED,
                        )
                    key_frame = cv2.addWeighted(key_frame, 1.0, blk, 0.5, 1)
                    for t in range(len(text)):
                        cv2.putText(
                            key_frame, text[t], coord[t], font, 0.5, (0, 0, 0), 1
                        )

            # save
            out.write(frame)

            if args.show:
                # show
                cv2.imshow("key-frame detection", frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


@torch.no_grad()
def detect_semi_stream(args, d_cfg, model, device, transform, class_names):
    # path to save
    save_path = os.path.join(args.save_folder, "ava_video")
    os.makedirs(save_path, exist_ok=True)

    # path to video
    path_to_video = os.path.join(d_cfg["data_root"], "videos_15min", args.video)

    # video
    video = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    save_size = (640, 480)
    save_name = os.path.join(save_path, "detection.avi")
    fps = 15.0
    out = cv2.VideoWriter(save_name, fourcc, fps, save_size)

    # run
    video_clip = []
    while True:
        ret, frame = video.read()

        if ret:
            # to PIL image
            frame_pil = Image.fromarray(frame.astype(np.uint8))

            # prepare
            if len(video_clip) <= 0:
                for _ in range(d_cfg["len_clip"]):
                    video_clip.append(frame_pil)

            video_clip.append(frame_pil)
            del video_clip[0]

            # orig size
            orig_h, orig_w = frame.shape[:2]

            # transform
            x, _ = transform(video_clip)
            # List [T, 3, H, W] -> [T, 3, H, W]
            x = torch.stack(x)
            x = x.unsqueeze(0).to(device)  # [B, T, 3, H, W], B=1

            t0 = time.time()
            # inference
            out_bboxes = model(x)
            print("inference time ", time.time() - t0, "s")

            # visualize detection results
            for bbox in out_bboxes:
                x1, y1, x2, y2 = bbox[:4]
                cls_out = bbox[4:]

                # rescale bbox
                x1, x2 = int(x1 * orig_w), int(x2 * orig_w)
                y1, y2 = int(y1 * orig_h), int(y2 * orig_h)

                cls_scores = np.array(cls_out)
                indices = np.where(cls_scores > args.vis_thresh)
                scores = cls_scores[indices]
                indices = list(indices[0])
                scores = list(scores)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if len(scores) > 0:
                    blk = np.zeros(frame.shape, np.uint8)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    coord = []
                    text = []
                    text_size = []

                    for _, cls_ind in enumerate(indices):
                        text.append(
                            "[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind])
                        )
                        text_size.append(
                            cv2.getTextSize(
                                text[-1], font, fontScale=0.25, thickness=1
                            )[0]
                        )
                        coord.append((x1 + 3, y1 + 7 + 10 * _))
                        cv2.rectangle(
                            blk,
                            (coord[-1][0] - 1, coord[-1][1] - 6),
                            (
                                coord[-1][0] + text_size[-1][0] + 1,
                                coord[-1][1] + text_size[-1][1] - 4,
                            ),
                            (0, 255, 0),
                            cv2.FILLED,
                        )
                    frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)
                    for t in range(len(text)):
                        cv2.putText(frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)

            # save
            out.write(frame)

            if args.show:
                # show
                cv2.imshow("key-frame detection", frame)
                cv2.waitKey(1)

        else:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    # cuda
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg["label_map"]
    num_classes = 80

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg["test_size"],
        pixel_mean=d_cfg["pixel_mean"],
        pixel_std=d_cfg["pixel_std"],
    )

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=False,
    )

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # inference mode
    model.set_inference_mode(args.inf_mode)

    if args.inf_mode == "clip":
        # run
        detect_clip(
            args=args,
            d_cfg=d_cfg,
            model=model,
            device=device,
            transform=basetransform,
            class_names=class_names,
        )
    elif args.inf_mode == "stream":
        # run
        detect_stream(
            args=args,
            d_cfg=d_cfg,
            model=model,
            device=device,
            transform=basetransform,
            class_names=class_names,
        )
    elif args.inf_mode == "semi_stream":
        # run
        detect_semi_stream(
            args=args,
            d_cfg=d_cfg,
            model=model,
            device=device,
            transform=basetransform,
            class_names=class_names,
        )
