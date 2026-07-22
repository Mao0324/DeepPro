"""
Author: Benny
Date: Nov 2019
"""
import argparse
import gc
import os
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from data_utils.TestDataLoader import TestIRSeqDataLoader
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import random
import torch.nn.functional as F

try:
    import swanlab
except ImportError:
    swanlab = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def seed_everything(seed=46):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multiprocessing_loader_options(worker_count, prefetch_factor):
    """Return DataLoader options that are valid with and without workers."""
    options = {'num_workers': worker_count}
    if worker_count > 0:
        options.update({
            'persistent_workers': True,
            'prefetch_factor': prefetch_factor,
        })
    return options


def binary_segmentation_metrics(
    true_positive,
    predicted_positive,
    target_positive,
):
    """Return micro-averaged pixel IoU, precision, recall and F1."""
    true_positive = true_positive.to(torch.float64)
    predicted_positive = predicted_positive.to(torch.float64)
    target_positive = target_positive.to(torch.float64)
    union = predicted_positive + target_positive - true_positive
    metrics = torch.stack((
        true_positive / union.clamp_min(1),
        true_positive / predicted_positive.clamp_min(1),
        true_positive / target_positive.clamp_min(1),
        2 * true_positive / (predicted_positive + target_positive).clamp_min(1),
    ))
    return tuple(metrics.tolist())


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='DeepPro-Plus', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--gpu_num', type=int, default=1, help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--datapath', type=str, default='./datasets/NUDT-MIRSDT')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, NUDT-MIRSDT-HiNo, '
                                            'RGB-T, SatVideoIRSDT, IRDST-simulation, IRSatVideo-LEO]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--savepath', type=str, default='./log/', help='Save path [default: ./log/]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--seqlen', type=int, default=40, help='Frame number as an input [default: 100]')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch Size for train generator [default: 128, 72]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--sample_rate', type=float, default=0.1, help='Sampling rate for training [default: 0.1(NUDT-MIRSDT), '
                                                                     '0.03(IRDST), 0.05(RGB-T), 0.04(SatVideoIRSDT)]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--threshold_eval', type=float, default=0.5, help='Threshold in evaluation [default: 0.5]')
    parser.add_argument('--train_workers', type=int, default=8,
                        help='Persistent DataLoader workers used for training [default: 8]')
    parser.add_argument('--val_workers', type=int, default=4,
                        help='Persistent DataLoader workers used for validation [default: 4]')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Batches prefetched by each DataLoader worker [default: 2]')
    parser.add_argument('--use_swanlab', type=int, default=1, choices=[0, 1], help='Use SwanLab logging [default: 1]')
    parser.add_argument('--swanlab_project', type=str, default='DeepPro', help='SwanLab project name')
    parser.add_argument("--spatial_ckpt", type=str, default="")
    parser.add_argument("--st_ckpt", type=str, default="")
    parser.add_argument("--freeze_pretrained", type=int, default=1)

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    def release_cuda_memory(stage):
        """Release unreferenced CUDA blocks at train/eval phase boundaries."""
        gc.collect()
        if not torch.cuda.is_available():
            return
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        allocated = sum(
            torch.cuda.memory_allocated(device_id)
            for device_id in range(torch.cuda.device_count())
        ) / (1024 ** 3)
        reserved = sum(
            torch.cuda.memory_reserved(device_id)
            for device_id in range(torch.cuda.device_count())
        ) / (1024 ** 3)
        log_string(
            'CUDA memory after %s: allocated=%.3f GiB, reserved=%.3f GiB'
            % (stage, allocated, reserved)
        )

    '''HYPER PARAMETER'''
    if args.train_workers < 0 or args.val_workers < 0:
        raise ValueError('DataLoader worker counts must be non-negative.')
    if args.prefetch_factor <= 0:
        raise ValueError('prefetch_factor must be positive.')
    if args.gpu_num == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(args.savepath)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        args.log_dir = args.dataset + '__' + timestr + '__SoftLoUloss_' + args.model + '_DataL' + str(args.seqlen)
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    swanlab_run = None
    if args.use_swanlab:
        if swanlab is None:
            log_string('SwanLab is not installed. Skip SwanLab logging.')
        else:
            try:
                swanlab_run = swanlab.init(
                    project=args.swanlab_project,
                    experiment_name=args.log_dir,
                    config=vars(args),
                )
            except Exception as e:
                log_string('SwanLab init failed: %s. Skip SwanLab logging.' % e)
                swanlab_run = None

    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = TrainIRSeqDataLoader(args.dataset, data_root=root, seq_len=SEQ_LEN, sample_rate=args.sample_rate,
                                         patch_size=args.patch_size, transform=None)  # sample_rate=0.1, 0.03, 0.05
    print("start loading test data ...")
    TEST_DATASET  = TestIRSeqDataLoader(args.dataset, data_root=root,  seq_len=SEQ_LEN, cat_len=int(SEQ_LEN*0.1), transform=None)
    VALIDATION_DATASET = TEST_DATASET.flatten_windows()

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x + int(time.time())),
        **multiprocessing_loader_options(
            args.train_workers,
            args.prefetch_factor,
        )
    )
    validationDataLoader = torch.utils.data.DataLoader(
        VALIDATION_DATASET,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        **multiprocessing_loader_options(
            args.val_workers,
            args.prefetch_factor,
        )
    )

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d sequences" % len(TEST_DATASET))
    log_string(
        "DataLoader workers: train=%d, validation=%d, prefetch_factor=%d"
        % (args.train_workers, args.val_workers, args.prefetch_factor)
    )

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('networks/models/%s.py' % args.model, str(experiment_dir))

    if "TDCSTA" in args.model:
        detector = MODEL.detector(
            NUM_CLASSES,
            SEQ_LEN,
            SEQ_LEN,
            spatial_ckpt=args.spatial_ckpt,
            st_ckpt=args.st_ckpt,
            freeze_pretrained=bool(args.freeze_pretrained),
        )
    else:
        detector = MODEL.detector(NUM_CLASSES, SEQ_LEN, SEQ_LEN)
    if args.gpu_num > 1:
        detector = torch.nn.DataParallel(detector)#, device_ids=list(np.arange(args.gpu_num)))
    detector = detector.cuda()
    # criterion = MODEL.bceloss().cuda()
    # criterion = MODEL.HAMloss().cuda()
    criterion = MODEL.SoftLoUloss().cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, detector.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, detector.parameters()),
            lr=args.learning_rate,
            momentum=0.9
        )

    best_iou = 0
    checkpoint = None
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch'] + 1
        if hasattr(detector, 'module'):
            detector.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_iou = checkpoint.get('class_avg_iou', 0)
        log_string('Use pretrain model')
    except Exception as e:
        log_string('No existing model or failed to load checkpoint: %s' % e)
        log_string('Starting training from scratch...')
        start_epoch = 0

    # load_state_dict copies checkpoint values into the live model/optimizer;
    # retaining the source dictionary would keep a duplicate set of tensors.
    del checkpoint
    release_cuda_memory('checkpoint loading')


    LEARNING_RATE_CLIP = 1e-5
    global_epoch = 0
    ## train
    for epoch in range(start_epoch, args.epoch):
        '''Train'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        num_batches = len(trainDataLoader)
        metric_device = next(detector.parameters()).device
        total_true_positive_mid = torch.zeros(
            (), device=metric_device, dtype=torch.int64
        )
        total_predicted_positive_mid = torch.zeros(
            (), device=metric_device, dtype=torch.int64
        )
        total_target_positive_mid = torch.zeros(
            (), device=metric_device, dtype=torch.int64
        )
        loss_sum = torch.zeros(
            (), device=metric_device, dtype=torch.float64
        )
        detector.train()

        for i, (images, targets) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad(set_to_none=True)
            #torch.autograd.set_detect_anomaly = True
            torch.autograd.set_detect_anomaly(True)
            images = images.float().cuda(non_blocking=True)
            targets = targets.float().cuda(non_blocking=True)

            _, seq_midpred = detector(images)

            loss = criterion(seq_midpred, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                midpred_choice = torch.sigmoid(seq_midpred.detach()).gt(
                    args.threshold_eval
                )
                batch_label = targets.gt(0)
                total_true_positive_mid += torch.logical_and(
                    midpred_choice,
                    batch_label,
                ).sum(dtype=torch.int64)
                total_predicted_positive_mid += midpred_choice.sum(
                    dtype=torch.int64
                )
                total_target_positive_mid += batch_label.sum(
                    dtype=torch.int64
                )
                loss_sum += loss.detach().to(torch.float64)
            # break

        # The last training batch and its gradients otherwise remain alive
        # throughout validation because Python blocks do not create a scope.
        optimizer.zero_grad(set_to_none=True)
        train_loss = (loss_sum / num_batches).item()
        train_iou, train_precision, train_recall, train_f1 = (
            binary_segmentation_metrics(
                total_true_positive_mid,
                total_predicted_positive_mid,
                total_target_positive_mid,
            )
        )

        del images, targets, seq_midpred, loss
        del midpred_choice, batch_label
        del total_true_positive_mid
        del total_predicted_positive_mid, total_target_positive_mid, loss_sum
        release_cuda_memory('training cleanup')

        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy (IoU) of prediction: %f' % train_iou)
        log_string('Training pixel precision: %f' % train_precision)
        log_string('Training pixel recall: %f' % train_recall)
        log_string('Training pixel F1: %f' % train_f1)
        if swanlab_run is not None:
            swanlab.log({
                'train/loss': train_loss,
                'train/iou': train_iou,
                'train/precision': train_precision,
                'train/recall': train_recall,
                'train/f1': train_f1,
                'train/lr': lr,
            }, step=epoch + 1)

        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epoch:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/epoch_' + str(epoch+1) + '_model.pth'
            log_string('Saving at %s' % savepath)
            if args.gpu_num > 1:
                state = {
                    'epoch': epoch,
                    'model_state_dict': detector.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            else:
                state = {
                    'epoch': epoch,
                    'model_state_dict': detector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(state, savepath)
            del state
            log_string('Saving model....')

        '''Evaluate'''
        with torch.inference_mode():
            num_batches = len(validationDataLoader)
            total_true_positive_mid = torch.zeros(
                (), device=metric_device, dtype=torch.int64
            )
            total_predicted_positive_mid = torch.zeros(
                (), device=metric_device, dtype=torch.int64
            )
            total_target_positive_mid = torch.zeros(
                (), device=metric_device, dtype=torch.int64
            )
            loss_g_sum = torch.zeros(
                (), device=metric_device, dtype=torch.float64
            )
            detector.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for images, targets, _, _ in tqdm(
                validationDataLoader,
                total=num_batches,
                smoothing=0.9,
            ):
                images = images.float().cuda(non_blocking=True)
                targets = targets.float().cuda(non_blocking=True)

                _, seq_midpred = detector(images)
                if seq_midpred.shape[-1] != targets.shape[-1]:
                    seq_midpred = F.interpolate(
                        seq_midpred,
                        size=targets.shape[-2:],
                    )

                loss_g_sum += criterion(seq_midpred, targets).to(torch.float64)
                pred_choice_mid = torch.sigmoid(seq_midpred).gt(
                    args.threshold_eval
                )
                batch_label = targets.gt(0)
                total_true_positive_mid += torch.logical_and(
                    pred_choice_mid,
                    batch_label,
                ).sum(dtype=torch.int64)
                total_predicted_positive_mid += pred_choice_mid.sum(
                    dtype=torch.int64
                )
                total_target_positive_mid += batch_label.sum(
                    dtype=torch.int64
                )

            mIoU_mid, eval_precision, eval_recall, eval_f1 = (
                binary_segmentation_metrics(
                    total_true_positive_mid,
                    total_predicted_positive_mid,
                    total_target_positive_mid,
                )
            )
            eval_loss = (loss_g_sum / num_batches).item()
            del images, targets, seq_midpred
            del pred_choice_mid, batch_label
            del total_true_positive_mid
            del total_predicted_positive_mid, total_target_positive_mid
            del loss_g_sum
            log_string('Eval mean loss: %f' % eval_loss)
            log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))
            log_string('Eval pixel precision: %f' % eval_precision)
            log_string('Eval pixel recall: %f' % eval_recall)
            log_string('Eval pixel F1: %f' % eval_f1)

            if mIoU_mid >= best_iou:
                best_iou = mIoU_mid
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                if args.gpu_num > 1:
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU_mid,
                        'model_state_dict': detector.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                else:
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU_mid,
                        'model_state_dict': detector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                torch.save(state, savepath)
                del state
                log_string('Saving model....')
            log_string('Best mIoU_mid: %f' % best_iou)
            if swanlab_run is not None:
                swanlab.log({
                    'eval/loss': eval_loss,
                    'eval/iou': mIoU_mid,
                    'eval/precision': eval_precision,
                    'eval/recall': eval_recall,
                    'eval/f1': eval_f1,
                    'eval/best_iou': best_iou,
                }, step=epoch + 1)

        release_cuda_memory('evaluation cleanup')
        global_epoch += 1

    if swanlab_run is not None and hasattr(swanlab, 'finish'):
        swanlab.finish()

    # main() is followed by a separate test.py process. Explicit teardown is
    # required so the parent process does not keep CUDA cache while the child
    # allocates its own model and inputs.
    del trainDataLoader, validationDataLoader
    del detector, criterion, optimizer
    release_cuda_memory('training shutdown')


def path_remake(path):
    return path.replace(' ', '\ ').replace('(', '\(').replace(')', '\)').replace('&', '\&')


if __name__ == '__main__':
    args = parse_args()
    main(args)

    os.system('python test.py --gpu %s --seqlen %d --datapath %s --dataset %s --log_dir %s' % (
            args.gpu, args.seqlen, path_remake(args.datapath), path_remake(args.dataset), path_remake(args.log_dir)))

