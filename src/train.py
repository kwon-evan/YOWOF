import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import datetime # noqa E402
import argparse  # noqa E402
import os  # noqa E402
import random  # noqa E402
import time  # noqa E402
from copy import deepcopy  # noqa E402

import numpy as np  # noqa E402
import torch  # noqa E402
import torch.backends.cudnn as cudnn  # noqa E402
import torch.amp as amp  # noqa E402
import torch.distributed as dist  # noqa E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa E402
from torch.optim.lr_scheduler import MultiStepLR  # noqa E402

from yowof.config import build_dataset_config, build_model_config  # noqa E402
from yowof.models.detector import build_model  # noqa E402
from yowof.utils import distributed_utils  # noqa E402
from yowof.utils.com_flops_params import FLOPs_and_Params  # noqa E402
from yowof.utils.misc import CollateFunc, build_dataloader, build_dataset  # noqa E402
from yowof.utils.solver.optimizer import build_optimizer  # noqa E402
from yowof.utils.solver.warmup_schedule import build_warmup  # noqa E402


def parse_args():
    parser = argparse.ArgumentParser(description="YOWOF")
    # CUDA
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda.")

    # Visualization
    parser.add_argument(
        "--tfboard", action="store_true", default=False, help="use tensorboard"
    )
    parser.add_argument(
        "--save_folder", default="weights/", type=str, help="path to save weight"
    )
    parser.add_argument(
        "--vis_data", action="store_true", default=False, help="use tensorboard"
    )

    # Mix precision training
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=False,
        help="Adopting mix precision training.",
    )

    # Evaluation
    parser.add_argument(
        "--eval",
        action="store_true",
        default=False,
        help="do evaluation during training.",
    )
    parser.add_argument(
        "--eval_epoch",
        default=1,
        type=int,
        help="after eval epoch, the model is evaluated on val dataset.",
    )
    parser.add_argument(
        "--save_dir",
        default="inference_results/",
        type=str,
        help="save inference results.",
    )

    # Model
    parser.add_argument(
        "-v",
        "--version",
        default="yowof-r18",
        type=str,
        help="build spatio-temporal action detector",
    )
    parser.add_argument(
        "--topk", default=50, type=int, help="topk candidates for evaluation"
    )
    parser.add_argument("-r", "--resume", default=None, type=str, help="keep training")

    # Dataset
    parser.add_argument(
        "-d", "--dataset", default="ucf24", help="ucf24, jhmdb21, ava_v2.2"
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers used in dataloading",
    )

    # DDP train
    parser.add_argument(
        "-dist",
        "--distributed",
        action="store_true",
        default=False,
        help="distributed training",
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--sybn", action="store_true", default=False, help="use sybn.")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    print("World size: {}".format(distributed_utils.get_world_size()))
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print("use cuda")
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # amp
    scaler = amp.GradScaler(enabled=args.fp16)

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(device, d_cfg, args, is_train=True)

    # dataloader
    each_gpu_batch_size = m_cfg["batch_size"] // distributed_utils.get_world_size()
    dataloader = build_dataloader(
        args, dataset, each_gpu_batch_size, CollateFunc(), is_train=True
    )

    # build model
    net = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        resume=args.resume,
    )
    model = net
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print("use SyncBatchNorm ...")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer
    base_lr = d_cfg["base_lr"]
    optimizer, start_epoch = build_optimizer(
        d_cfg, model_without_ddp, base_lr, args.resume
    )

    # lr scheduler
    lr_scheduler = MultiStepLR(
        optimizer=optimizer, milestones=d_cfg["lr_epoch"], gamma=d_cfg["lr_decay_ratio"]
    )

    # warmup scheduler
    warmup_scheduler = build_warmup(cfg=d_cfg, base_lr=base_lr)

    # training configuration
    max_epoch = d_cfg["max_epoch"]
    epoch_size = len(dataloader)
    warmup = True

    # Compute FLOPs and Params
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg["test_size"],
            len_clip=d_cfg["len_clip"],
            device=device,
        )
        del model_copy

    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # warmup
            if ni < d_cfg["wp_iter"] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg["wp_iter"] and warmup:
                # warmup is over
                print("Warmup is over")
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            # to device
            video_clips = video_clips.to(device)

            # inference
            if args.fp16:
                with torch.amp.autocast("cuda", enabled=args.fp16):
                    loss_dict = model(video_clips, targets=targets)
            else:
                loss_dict = model(video_clips, targets=targets)

            losses = loss_dict["losses"]
            losses = losses / m_cfg["accumulate"]

            # reduce
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print("loss is NAN !!")
                continue

            # Backward and Optimize
            if args.fp16:
                scaler.scale(losses).backward()

                # Optimize
                if ni % m_cfg["accumulate"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                # Backward
                losses.backward()

                # Optimize
                if ni % m_cfg["accumulate"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group["lr"] for param_group in optimizer.param_groups]
                # basic infor
                log = "[Epoch: {}/{}]".format(epoch + 1, max_epoch)
                log += "[Iter: {}/{}]".format(iter_i, epoch_size)
                log += "[lr: {:.2f}]".format(cur_lr[0])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += "[{}: {:.2f}]".format(k, loss_dict[k])

                # other infor
                log += "[time: {:.2f}]".format(t1 - t0)
                log += "[size: {}]".format(d_cfg["train_size"])

                # print log infor
                print(log, flush=True)

                t0 = time.time()

        lr_scheduler.step()

        # evaluation
        if epoch % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            model_eval = model_without_ddp
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print("No evaluator ... save model and go on training.")
                    print("Saving state, epoch: {}".format(epoch + 1))
                    weight_name = "{}_epoch_{}.pth".format(args.version, epoch + 1)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save(
                        {
                            "model": model_eval.state_dict(),
                            # 'optimizer': optimizer.state_dict(),
                            # 'lr_scheduler': lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

                else:
                    print("eval ...")
                    # set eval mode
                    model_eval.trainable = False
                    model_eval.eval()

                    # evaluate
                    evaluator.evaluate_frame_map(model_eval, epoch + 1)

                    # set train mode.
                    model_eval.trainable = True
                    model_eval.train()

                    # save model
                    print("Saving state, epoch:", epoch + 1)
                    weight_name = "{}_epoch_{}.pth".format(args.version, epoch + 1)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save(
                        {
                            "model": model_eval.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

                    # set train mode.
                    model_eval.trainable = True
                    model_eval.train()
                    if args.distributed:
                        model.module.set_inference_mode(mode="clip")
                    else:
                        model.set_inference_mode(mode="clip")

            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()


if __name__ == "__main__":
    train()
