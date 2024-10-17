#!/usr/bin/python
# encoding: utf-8

import os
import json
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        dataset="custom",
        img_size=224,
        transform=None,
        is_train=False,
        len_clip=16,
        sampling_rate=1,
    ):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train

        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate

        if self.is_train:
            self.split_list = "trainlist.txt"
        else:
            self.split_list = "testlist.txt"

        # load data
        with open(os.path.join(data_root, self.split_list), "r") as file:
            self.file_names = file.readlines()

        self.num_samples = len(self.file_names)
        self.path_to_video = None

        self.num_classes = 3  # FLAME, SMOKE, NORMAL

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        image_path = self.file_names[index].rstrip()

        # load a data
        frame_idx, video_clip, target = self.pull_item(image_path)

        return frame_idx, video_clip, target

    def pull_item(self, image_path):
        """load a data"""

        img_split = image_path.split("/")
        # image name
        img_info = img_split[-1].split(".")[0]
        vid_id, cls, place, img_id = img_info.split("_")
        img_id = int(img_id)

        # path to label
        label_path = os.path.join(
            self.data_root,
            *img_split[1:4],
            "02.라벨링데이터",
            *img_split[5:10],
            "JSON",
            img_info + ".json",
        )

        img_folder = os.path.join(self.data_root, *img_split[:-1])

        # frame numbers
        max_num = len(os.listdir(img_folder))

        # sampling rate
        if self.is_train:
            d = random.randint(1, 2)
        else:
            d = self.sampling_rate

        # load images
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i * d
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            path_tmp = os.path.join(
                self.data_root,
                *img_split[:-2],
                "JPG",
                f"{vid_id}_{cls}_{place}_{img_id_temp:05d}.jpg",
            )
            frame = Image.open(path_tmp).convert("RGB")
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

            # frame_id = img_split[1] + "_" + img_split[2] + "_" + img_split[3]
            frame_id = img_info

        # load an annotation
        if not os.path.getsize(label_path):
            raise Exception("no annotation file")

        # target = np.loadtxt(label_path)
        with open(label_path, "r") as f:
            annotation = json.load(f)

        ow, oh = annotation["image"]["width"], annotation["image"]["height"]
        target = []
        for i in annotation["annotations"]:
            x, y, w, h = i["bbox"]  # center x, center y, width, height
            x1, y1, x2, y2 = x, y, (x + w), (y + h)
            target.append([x1, y1, x2, y2, i["categories_id"]])


        # [x1, y1, x2, y2, label]
        target = np.array(target).reshape(-1, 5)

        # transform
        video_clip, target = self.transform(video_clip, target)
        # List [T, 3, H, W] -> [T, 3, H, W]
        video_clip = torch.stack(video_clip)

        # reformat target
        target = {
            "boxes": target[:, :4].float(),  # [N, 4]
            "labels": target[:, -1].long() - 1,  # [N,]
            "orig_size": torch.as_tensor([ow, oh]),
        }

        return frame_id, video_clip, target


if __name__ == "__main__":
    import cv2
    from yowof.dataset.transforms import Augmentation, BaseTransform

    # data_root = "D:/python_work/spatial-temporal_action_detection/dataset/ucf24"
    data_root = "/home/bom/바탕화면/datasets/089.화재 발생 예측 영상_고도화_영상 기반 화재 감시 및 발생 위치 탐지 데이터"
    dataset = "custom"
    is_train = True
    img_size = 224
    len_clip = 16
    trans_config = {
        "pixel_mean": [0.485, 0.456, 0.406],
        "pixel_std": [0.229, 0.224, 0.225],
        "jitter": 0.2,
        "hue": 0.1,
        "saturation": 1.5,
        "exposure": 1.5,
    }
    transform = Augmentation(
        img_size=img_size,
        pixel_mean=trans_config["pixel_mean"],
        pixel_std=trans_config["pixel_std"],
        jitter=trans_config["jitter"],
        saturation=trans_config["saturation"],
        exposure=trans_config["exposure"],
    )
    transform = BaseTransform(
        img_size, trans_config["pixel_mean"], trans_config["pixel_std"]
    )

    train_dataset = CustomDataset(
        data_root=data_root,
        dataset=dataset,
        img_size=img_size,
        transform=transform,
        is_train=is_train,
        len_clip=len_clip,
        sampling_rate=1,
    )

    print(len(train_dataset))
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[-1]

        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = (key_frame * trans_config['pixel_std'] + trans_config['pixel_mean']) * 255
        key_frame = key_frame.astype(np.uint8)
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        
        # # PIL show
        # image = Image.fromarray(image.astype(np.uint8))
        # image.show()

        # cv2 show
        # cv2.imshow('key frame', key_frame[..., (2, 1, 0)])
        # cv2.waitKey(0)

        print(frame_id, key_frame.shape, bboxes, labels)
