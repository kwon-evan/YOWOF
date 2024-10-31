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
        gt_folder=None,
        save_path=None,
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

        self.metric = MeanAveragePrecision(
            iou_thresholds=[0.5, 0.75, 0.9], class_metrics=True
        )

    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        print("Metric: Frame mAP")
        epoch_size = len(self.testset)

        # init model
        model.set_inference_mode(mode="stream")

        # inference
        total_map = 0
        for iter_i, (frame_id, video_clip, target) in enumerate(self.testset):
            video_clip = video_clip.unsqueeze(0).to(self.device)  # [B, T, 3, H, W], B=1

            scores, labels, bboxes = model(video_clip)

            # rescale bboxes
            orig_size = target["orig_size"].tolist()
            bboxes = rescale_bboxes(bboxes, orig_size)

            preds = {
                "boxes": bboxes,
                "scores": scores,
                "labels": labels,
            }

            map_dict = self.metric(preds, target)
            total_map += map_dict["map"]

            print(
                f"Epoch {epoch}: {iter_i}/{epoch_size} | {frame_id} | mAP: {map_dict['map']:.4f}"
            )
            print(map_dict)

        avg_map = total_map / epoch_size
        print(f"Epoch {epoch} | mAP: {avg_map:.4f}")
        print("-----------------------------------")

    def evaluate_video_map(self, model, epoch=1):
        raise NotImplementedError


if __name__ == "__main__":
    pass
