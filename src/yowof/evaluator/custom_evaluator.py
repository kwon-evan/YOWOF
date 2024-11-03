import time
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yowof.dataset.custom import CustomDataset
from yowof.utils.box_ops import rescale_bboxes


class Custom_Evaluator(object):
    def __init__(
        self,
        device=None,
        data_root=None,
        dataset="ucf24",
        model_name="yowof-r18",
        img_size=320,
        len_clip=1,
        conf_thresh=0.01,
        iou_thresh=0.5,
        transform=None,
    ):
        self.device = device
        self.data_root = data_root
        self.dataset = dataset
        self.model_name = model_name
        self.img_size = img_size
        self.len_clip = len_clip
        self.transform = transform
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # dataset
        self.testset = CustomDataset(
            data_root=data_root,
            dataset=dataset,
            img_size=img_size,
            transform=transform,
            is_train=False,
            len_clip=len_clip,
            sampling_rate=1,
        )
        self.num_classes = self.testset.num_classes
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            shuffle=False,
            collate_fn=None,
            num_workers=8,
            batch_size=8,
            drop_last=False,
            pin_memory=True,
        )

        self.metric = MeanAveragePrecision(
            iou_thresholds=[0.5, 0.75, 0.9],
            class_metrics=True,
            sync_on_compute=False,
            dist_sync_on_step=True,
        )

    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        print("Metric: Frame mAP")
        epoch_size = len(self.dataloader)

        # init model
        model.set_inference_mode(mode="stream")

        # inference
        tic = time.time()
        for iter_i, (frame_ids, video_clips, targets) in enumerate(self.dataloader):
            video_clips = video_clips.to(self.device)

            scores, labels, bboxes = model(video_clips)

            # rescale bboxes
            orig_size = targets["orig_size"][0].tolist()
            bboxes = rescale_bboxes(bboxes, orig_size)
            pred = [
                dict(
                    boxes=torch.from_numpy(bboxes),
                    scores=torch.from_numpy(scores),
                    labels=torch.from_numpy(labels),
                )
            ]
            targets["boxes"] = targets["boxes"].squeeze()
            targets["labels"] = targets["labels"].squeeze()

            self.metric.update(pred, [targets])
            if iter_i % 10 == 0:
                toc = time.time()
                print(f"Epoch {epoch}: {iter_i}/{epoch_size} | time: {toc - tic:.4f}")
                tic = toc

        map_dict = self.metric.compute()
        print(f"Epoch {epoch} | mAP: {map_dict['map']:.4f}")
        print(map_dict)
        print("-----------------------------------")

    def evaluate_video_map(self, model, epoch=1):
        raise NotImplementedError


if __name__ == "__main__":
    pass

