import os
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from utils.box_ops import rescale_bboxes
from utils.box_ops import rescale_bboxes

from .cal_mAP import get_mAP
from .utils import bbox_iou


class UCF_JHMDB_Evaluator(object):
    def __init__(self,
                 device=None,
                 data_root=None,
                 dataset='ucf24',
                 model_name='yowo',
                 img_size=224,
                 len_clip=1,
                 conf_thresh=0.01,
                 iou_thresh=0.5,
                 transform=None,
                 redo=False,
                 gt_folder=None,
                 dt_folder=None,
                 save_path=None):
        self.device = device
        self.data_root = data_root
        self.dataset = dataset
        self.model_name = model_name
        self.img_size = img_size
        self.len_clip = len_clip
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.redo = redo
        self.gt_folder = gt_folder
        self.dt_folder = dt_folder
        self.save_path = save_path

        # dataset
        self.testset = UCF_JHMDB_Dataset(
            data_root=data_root,
            dataset=dataset,
            img_size=img_size,
            transform=transform,
            is_train=False,
            len_clip=len_clip,
            sampling_rate=1)
        self.num_classes = self.testset.num_classes


    @torch.no_grad()
    def evaluate_accu_recall(self, model, epoch=1):
        # number of groundtruth
        total_num_gts = 0
        proposals   = 0.0
        correct     = 0.0
        fscore = 0.0

        correct_classification = 0.0
        total_detected = 0.0
        eps = 1e-5

        epoch_size = len(self.testset)

        # initalize model
        model.initialization = True
        model.set_inference_mode(mode='stream')

        # inference
        prev_frame_id = ''
        for iter_i, (frame_id, video_clip, target) in enumerate(self.testset):
            # orignal frame size
            orig_size = target['orig_size']  # width, height

            # ex: frame_id = Basketball_v_Basketball_g01_c01_00048.txt
            if iter_i == 0:
                prev_frame_id = frame_id[:-10]
                model.initialization = True

            if frame_id[:-10] != prev_frame_id:
                # a new video
                prev_frame_id = frame_id[:-10]
                model.initialization = True

            # prepare
            video_clip = video_clip.unsqueeze(0).to(self.device) # [B, 3, T, H, W], B=1

            with torch.no_grad():
                # inference
                scores, labels, bboxes = model(video_clip)

                # rescale bbox
                orig_size = target['orig_size']
                bboxes = rescale_bboxes(bboxes, orig_size)

                if not os.path.exists('results'):
                    os.mkdir('results')

                if self.dataset == 'ucf24':
                    detection_path = os.path.join('results', 'ucf_detections', self.model_name, 'detections_' + str(epoch), frame_id)
                    current_dir = os.path.join('results', 'ucf_detections',  self.model_name, 'detections_' + str(epoch))
                    if not os.path.exists('results/ucf_detections/'):
                        os.mkdir('results/ucf_detections/')
                    if not os.path.exists('results/ucf_detections/'+self.model_name):
                        os.mkdir('results/ucf_detections/'+self.model_name)
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)
                else:
                    detection_path = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch), frame_id)
                    current_dir = os.path.join('results', 'jhmdb_detections',  self.model_name, 'detections_' + str(epoch))
                    if not os.path.exists('results/jhmdb_detections/'):
                        os.mkdir('results/jhmdb_detections/')
                    if not os.path.exists('results/jhmdb_detections/'+self.model_name):
                        os.mkdir('results/jhmdb_detections/'+self.model_name)
                    if not os.path.exists(current_dir):
                        os.mkdir(current_dir)

                with open(detection_path, 'w+') as f_detect:
                    for score, label, bbox in zip(scores, labels, bboxes):
                        x1 = round(bbox[0])
                        y1 = round(bbox[1])
                        x2 = round(bbox[2])
                        y2 = round(bbox[3])
                        cls_id = int(label) + 1

                        f_detect.write(
                            str(cls_id) + ' ' + str(score) + ' ' \
                                + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                tgt_bboxes = target['boxes'].numpy()
                tgt_labels = target['labels'].numpy()
                ow, oh = target['orig_size']
                num_gts = tgt_bboxes.shape[0]

                # count number of total groundtruth
                total_num_gts += num_gts

                pred_list = [] # LIST OF CONFIDENT BOX INDICES
                for i in range(len(scores)):
                    score = scores[i]
                    if score > self.conf_thresh:
                        proposals += 1
                        pred_list.append(i)

                for i in range(num_gts):
                    tgt_bbox = tgt_bboxes[i]
                    tgt_label = tgt_labels[i]
                    # rescale groundtruth bbox
                    tgt_bbox[[0, 2]] *= ow
                    tgt_bbox[[1, 3]] *= oh

                    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_bbox
                    box_gt = [tgt_x1, tgt_y1, tgt_x2, tgt_y2, 1.0, 1.0, tgt_label]
                    best_iou = 0
                    best_j = -1

                    for j in pred_list: # ITERATE THROUGH ONLY CONFIDENT BOXES
                        iou = bbox_iou(box_gt, bboxes[j], x1y1x2y2=True)
                        if iou > best_iou:
                            best_j = j
                            best_iou = iou

                    if best_iou > self.iou_thresh:
                        total_detected += 1
                        # print(labels[best_j], tgt_label)
                        if int(labels[best_j]) == int(tgt_label):
                            correct_classification += 1

                    if best_iou > self.iou_thresh and int(labels[best_j]) == int(tgt_label):
                        correct += 1

                precision = 1.0 * correct / (proposals + eps)
                recall = 1.0 * correct / (total_num_gts + eps)
                fscore = 2.0 * precision * recall / (precision + recall + eps)

                if iter_i % 1000 == 0:
                    log_info = "[%d / %d] precision: %f, recall: %f, fscore: %f" % (iter_i, epoch_size, precision, recall, fscore)
                    print(log_info, flush=True)

        classification_accuracy = 1.0 * correct_classification / (total_detected + eps)
        locolization_recall = 1.0 * total_detected / (total_num_gts + eps)

        print("Classification accuracy: %.3f" % classification_accuracy)
        print("Locolization recall: %.3f" % locolization_recall)

        model.set_inference_mode(mode='clip')

        return classification_accuracy, locolization_recall, current_dir


    @torch.no_grad()
    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):
        if self.redo:
            (
                classification_accuracy,
                locolization_recall,
                current_dir
            ) = self.evaluate_accu_recall(model, epoch)

            result_path = current_dir
        else:
            result_path = self.dt_folder

        print('calculating Frame mAP ...')
        metric_list = get_mAP(self.gt_folder, result_path, self.iou_thresh,
                              self.save_path, self.dataset, show_pr_curve)
        for metric in metric_list:
            print(metric)



if __name__ == "__main__":
    pass
