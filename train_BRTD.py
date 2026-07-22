"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from data_utils.TestDataLoader import TestIRSeqDataLoader
from networks.losses import build_segmentation_loss
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
import subprocess
import json
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from sequence_utils import SequenceAccumulator, frame_range_length
from runtime_utils import (
    all_reduce_sum,
    atomic_torch_save,
    broadcast_object,
    distributed_barrier,
    finalize_distributed,
    initialize_distributed,
    launch_with_torchrun_if_needed,
    load_checkpoint,
    move_optimizer_state,
    parse_visible_devices,
    unwrap_model,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'networks/models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def seed_everything(seed=46, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clean_state_dict(state_dict):
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def evaluate_model(detector, criterion, dataset, args, device):
    """Overlap-aware GPU validation without per-window host synchronization."""
    detector.eval()
    metric_counts = torch.zeros(2, device=device, dtype=torch.int64)
    loss_stats = torch.zeros(2, device=device, dtype=torch.float64)
    eval_bar = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc='Eval',
        smoothing=0.9,
        ascii=True,
        ncols=100,
        mininterval=0.5,
        file=sys.stdout,
    )
    with torch.inference_mode():
        for _, sequence_dataset in eval_bar:
            sequence_loader = torch.utils.data.DataLoader(
                sequence_dataset,
                batch_size=1,
                shuffle=False,
            )
            accumulator = SequenceAccumulator()
            for images, targets, _centroids, first_end in sequence_loader:
                images = images.float().to(device, non_blocking=True)
                targets = targets.float().to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=bool(args.eval_amp)):
                    sequence_features, prediction = detector(images)
                    del sequence_features
                    if prediction.shape[-2:] != targets.shape[-2:]:
                        prediction = F.interpolate(
                            prediction,
                            size=targets.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    valid_length = frame_range_length(first_end)
                    window_loss = criterion(
                        prediction[:, :valid_length],
                        targets[:, :valid_length],
                    )
                loss_stats[0] += window_loss.to(torch.float64)
                loss_stats[1] += 1
                accumulator.add(
                    torch.sigmoid(prediction[:, :valid_length]),
                    targets[:, :valid_length],
                    first_end,
                )
                del images, targets, prediction, window_loss, _centroids

            prediction = accumulator.predictions.gt(args.threshold_eval)
            target = accumulator.targets.gt(0)
            metric_counts[0] += torch.logical_and(
                prediction, target
            ).sum(dtype=torch.int64)
            metric_counts[1] += torch.logical_or(
                prediction, target
            ).sum(dtype=torch.int64)
            del accumulator, prediction, target

    mean_iou = (
        metric_counts[0].to(torch.float64)
        / metric_counts[1].clamp_min(1).to(torch.float64)
    ).item()
    mean_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
    return mean_loss, mean_iou


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='DeepPro-Plus', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size during training [default: 1]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument(
        '--gpu', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
        help='visible GPU device(s); defaults to the existing CUDA_VISIBLE_DEVICES value',
    )
    parser.add_argument('--gpu_num', type=int, default=1, help='GPU to use')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--datapath', type=str, default='./datasets/NUDT-MIRSDT')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT', help='dataset name [default: NUDT-MIRSDT, NUDT-MIRSDT-HiNo, '
                                            'RGB-T, SatVideoIRSDT, SatVideoIRSDT_v1, IRDST-simulation, IRSatVideo-LEO]')
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
    parser.add_argument('--use_swanlab', type=int, default=0, choices=[0, 1], help='Use SwanLab logging [default: 0]')
    parser.add_argument('--swanlab_project', type=str, default='DeepPro', help='SwanLab project name')
    parser.add_argument('--seed', type=int, default=46, help='Random seed')
    parser.add_argument('--base_ckpt', type=str, default='', help='DeepPro-Plus checkpoint used to initialize BRTD')
    parser.add_argument('--base_lr_mult', type=float, default=0.1, help='Learning-rate multiplier for pretrained layers')
    parser.add_argument('--brtd_use_background', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_adaptive_tdc', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_use_gate', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_zero_init', type=int, default=1, choices=[0, 1])
    parser.add_argument('--eval_chunk_rows', type=int, default=0,
                        help='Rows per exact streaming decoder chunk in eval; 0 disables chunking')
    parser.add_argument('--run_test_after_train', type=int, default=1, choices=[0, 1])
    parser.add_argument('--amp', type=int, default=0, choices=[0, 1],
                        help='Use CUDA automatic mixed precision [default: 0]')
    parser.add_argument('--eval_amp', type=int, default=0, choices=[0, 1],
                        help='Use mixed precision during validation/testing [default: 0]')
    parser.add_argument('--isolated_eval', type=int, default=1, choices=[0, 1],
                        help='Run validation in a fresh subprocess [default: 1]')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Accumulate gradients across this many batches [default: 1]')
    parser.add_argument('--train_workers', type=int, default=4,
                        help='Total training workers split across DDP ranks')
    parser.add_argument('--deterministic', type=int, default=0, choices=[0, 1])
    parser.add_argument('--resume', choices=['auto', 'never'], default='auto')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument("--spatial_ckpt", type=str, default="")
    parser.add_argument("--st_ckpt", type=str, default="")
    parser.add_argument("--freeze_pretrained", type=int, default=1)

    return parser.parse_args()


def main(args):
    if args.accumulation_steps < 1:
        raise ValueError('--accumulation_steps must be at least 1.')
    if args.batch_size <= 0 or args.gpu_num <= 0 or args.train_workers < 0:
        raise ValueError('batch_size/gpu_num must be positive and workers non-negative.')
    runtime = initialize_distributed(args.gpu, args.gpu_num)
    if args.batch_size % runtime.world_size:
        finalize_distributed(runtime)
        raise ValueError(
            'Global --batch_size must be divisible by DDP world size.'
        )
    seed_everything(
        args.seed + runtime.rank,
        deterministic=bool(args.deterministic),
    )

    if args.log_dir is None and runtime.is_main:
        timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        args.log_dir = (
            args.dataset + '__' + timestr + '__SoftLoUloss_'
            + args.model + '_DataL' + str(args.seqlen)
        )
    args.log_dir = broadcast_object(args.log_dir, runtime)
    args.savepath = str(Path(args.savepath).expanduser().resolve())
    experiment_root = Path(args.savepath) / 'sem_seg'
    experiment_dir = (experiment_root / args.log_dir).resolve()
    try:
        experiment_dir.relative_to(experiment_root)
    except ValueError as error:
        finalize_distributed(runtime)
        raise ValueError(
            '--log_dir must name an experiment under %s.' % experiment_root
        ) from error
    checkpoints_dir = experiment_dir / 'checkpoints'
    log_dir = experiment_dir / 'logs'

    resume_path = None
    setup_error = None
    if runtime.is_main:
        try:
            preexisting = experiment_dir.exists() and any(experiment_dir.iterdir())
            if args.resume_checkpoint:
                candidate = Path(args.resume_checkpoint).expanduser().resolve()
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                resume_path = str(candidate)
            elif args.resume == 'auto':
                for filename in ('latest_model.pth', 'best_model.pth'):
                    candidate = checkpoints_dir / filename
                    if candidate.is_file():
                        resume_path = str(candidate)
                        break
                if resume_path is None and preexisting:
                    raise RuntimeError(
                        'Non-empty experiment has no valid resume checkpoint: %s'
                        % experiment_dir
                    )
            elif preexisting:
                raise RuntimeError(
                    '--resume never refuses to overwrite %s' % experiment_dir
                )
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            setup_error = '%s: %s' % (type(error).__name__, error)
    setup_error = broadcast_object(setup_error, runtime)
    resume_path = broadcast_object(resume_path, runtime)
    if setup_error:
        finalize_distributed(runtime)
        raise RuntimeError(setup_error)
    distributed_barrier(runtime)

    logger = logging.getLogger("Model-BRTD-rank%d" % runtime.rank)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    if runtime.is_main:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_dir / ('%s.txt' % args.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    def log_string(message):
        if runtime.is_main:
            logger.info(message)
            print(message, flush=True)

    log_string('PARAMETER ...')
    log_string(args)

    swanlab_run = None
    swanlab_module = None
    if args.use_swanlab and runtime.is_main:
        try:
            import swanlab as swanlab_module
        except ImportError:
            log_string('SwanLab is not installed. Skip SwanLab logging.')
        if swanlab_module is not None:
            try:
                swanlab_run = swanlab_module.init(
                    project=args.swanlab_project,
                    experiment_name=args.log_dir,
                    config=vars(args),
                )
            except Exception as e:
                log_string('SwanLab init failed: %s. Skip SwanLab logging.' % e)
                swanlab_run = None

    args.datapath = str(Path(args.datapath).expanduser().resolve())
    root = args.datapath
    NUM_CLASSES = 1
    SEQ_LEN = args.seqlen
    local_batch_size = args.batch_size // runtime.world_size
    train_workers = (
        0 if args.train_workers == 0 else
        max(1, (args.train_workers + runtime.world_size - 1) // runtime.world_size)
    )

    log_string("start loading training data ...")
    TRAIN_DATASET = TrainIRSeqDataLoader(
        args.dataset,
        data_root=root,
        seq_len=SEQ_LEN,
        sample_rate=args.sample_rate,
        patch_size=args.patch_size,
        transform=None,
    )
    TEST_DATASET = None
    if runtime.is_main and (not args.isolated_eval or runtime.distributed):
        log_string("start loading test data ...")
        TEST_DATASET = TestIRSeqDataLoader(
            args.dataset,
            data_root=root,
            seq_len=SEQ_LEN,
            cat_len=int(SEQ_LEN * 0.1),
            transform=None,
        )

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed + runtime.rank)
    train_sampler = None
    if runtime.distributed:
        train_sampler = DistributedSampler(
            TRAIN_DATASET,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=local_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=train_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    if len(trainDataLoader) == 0:
        finalize_distributed(runtime)
        raise RuntimeError('Training loader has no full global batch.')

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    if TEST_DATASET is not None:
        log_string("The number of test data is: %d sequences" % len(TEST_DATASET))
    log_string(
        'DDP world_size=%d, global_batch=%d, per_rank_batch=%d, workers/rank=%d'
        % (runtime.world_size, args.batch_size, local_batch_size, train_workers)
    )

    '''MODEL LOADING'''
    models_dir = (Path(ROOT_DIR) / 'networks' / 'models').resolve()
    model_source = (models_dir / ('%s.py' % args.model)).resolve()
    if model_source.parent != models_dir or not model_source.is_file():
        finalize_distributed(runtime)
        raise ValueError('Unknown or unsafe model name: %s' % args.model)
    MODEL = importlib.import_module(args.model)
    if runtime.is_main:
        model_snapshot = experiment_dir / model_source.name
        if not model_snapshot.exists():
            shutil.copy2(model_source, model_snapshot)
    checkpoint_model_config = {}

    if "BRTD" in args.model:
        checkpoint_model_config = {
            'use_background': bool(args.brtd_use_background),
            'adaptive_tdc': bool(args.brtd_adaptive_tdc),
            'use_gate': bool(args.brtd_use_gate),
            'zero_init': bool(args.brtd_zero_init),
            'eval_chunk_rows': args.eval_chunk_rows,
        }
        if runtime.is_main:
            adapter_source = Path(ROOT_DIR) / 'networks' / 'layers' / 'brtd_adapter.py'
            adapter_snapshot = experiment_dir / adapter_source.name
            if not adapter_snapshot.exists():
                shutil.copy2(adapter_source, adapter_snapshot)
        detector = MODEL.detector(
            NUM_CLASSES,
            SEQ_LEN,
            SEQ_LEN,
            **checkpoint_model_config,
        )
    elif "TDCSTA" in args.model:
        detector = MODEL.detector(
            NUM_CLASSES,
            SEQ_LEN,
            SEQ_LEN,
            spatial_ckpt=args.spatial_ckpt,
            st_ckpt=args.st_ckpt,
            freeze_pretrained=bool(args.freeze_pretrained),
        )
    else:
        if args.model == 'DeepPro-Plus':
            checkpoint_model_config = {
                'eval_chunk_rows': args.eval_chunk_rows,
            }
        detector = MODEL.detector(
            NUM_CLASSES,
            SEQ_LEN,
            SEQ_LEN,
            **checkpoint_model_config,
        )

    distributed_barrier(runtime)
    if args.base_ckpt and resume_path is None:
        checkpoint = load_checkpoint(args.base_ckpt, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        incompatible = detector.load_state_dict(
            clean_state_dict(state_dict),
            strict=False,
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith('brtd.')
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                'Base checkpoint does not match DeepPro-Plus. '
                'Missing: %s; unexpected: %s'
                % (invalid_missing, incompatible.unexpected_keys)
            )
        log_string(
            'Initialized DeepPro-Plus backbone from %s; new BRTD keys: %d'
            % (args.base_ckpt, len(incompatible.missing_keys))
        )
        del checkpoint, state_dict, incompatible

    detector = detector.to(runtime.device)
    if runtime.distributed:
        detector = DistributedDataParallel(
            detector,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
        )
    # criterion = MODEL.bceloss().cuda()
    # criterion = MODEL.HAMloss().cuda()
    criterion = build_segmentation_loss('soft_iou').to(runtime.device)

    if "BRTD" in args.model:
        base_parameters = []
        adapter_parameters = []
        for name, parameter in detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'brtd.' in name:
                adapter_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        parameter_groups = [
            {
                'params': base_parameters,
                'lr': args.learning_rate * args.base_lr_mult,
                'lr_scale': args.base_lr_mult,
            },
            {
                'params': adapter_parameters,
                'lr': args.learning_rate,
                'lr_scale': 1.0,
            },
        ]
    else:
        parameter_groups = [{
            'params': list(filter(lambda p: p.requires_grad, detector.parameters())),
            'lr': args.learning_rate,
            'lr_scale': 1.0,
        }]

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            parameter_groups,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=0.9
        )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    best_iou = 0
    start_epoch = 0
    if resume_path is not None:
        try:
            checkpoint = load_checkpoint(resume_path, map_location='cpu')
            checkpoint_model = checkpoint.get('model_name')
            if checkpoint_model is not None and checkpoint_model != args.model:
                raise ValueError(
                    'Checkpoint model %s does not match %s.'
                    % (checkpoint_model, args.model)
                )
            unwrap_model(detector).load_state_dict(
                clean_state_dict(checkpoint['model_state_dict']),
                strict=True,
            )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            move_optimizer_state(optimizer, runtime.device)
            if checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = int(checkpoint['epoch']) + 1
            best_iou = float(checkpoint.get('class_avg_iou', 0.0))
            log_string('Resume training from %s' % resume_path)
            del checkpoint
        except Exception as error:
            finalize_distributed(runtime)
            raise RuntimeError(
                'Checkpoint recovery failed; refusing to overwrite %s: %s'
                % (experiment_dir, error)
            ) from error
    else:
        log_string('Starting training from scratch...')


    LEARNING_RATE_CLIP = 1e-5
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d/%s ****' % (epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        num_batches = len(trainDataLoader)
        metric_counts = torch.zeros(2, device=runtime.device, dtype=torch.int64)
        loss_stats = torch.zeros(2, device=runtime.device, dtype=torch.float64)
        detector.train()
        optimizer.zero_grad(set_to_none=True)

        # for i, (images, targets) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        train_bar = tqdm(
            enumerate(trainDataLoader),
            total=len(trainDataLoader),
            desc='Train %03d/%03d' % (epoch + 1, args.epoch),
            smoothing=0.9,
            ascii=True,
            dynamic_ncols=False,
            ncols=100,
            mininterval=0.5,
            leave=True,
            file=sys.stdout,
            disable=not runtime.is_main,
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar:30}| '
                '{n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

        for i, (images, targets) in train_bar:
            #torch.autograd.set_detect_anomaly = True
            images = images.float().to(runtime.device, non_blocking=True)
            targets = targets.float().to(runtime.device, non_blocking=True)
            group_start = (i // args.accumulation_steps) * args.accumulation_steps
            group_size = min(
                args.accumulation_steps,
                num_batches - group_start,
            )
            should_step = (
                (i + 1) % args.accumulation_steps == 0
                or i + 1 == num_batches
            )
            synchronization = (
                detector.no_sync()
                if runtime.distributed and not should_step
                else nullcontext()
            )
            with synchronization:
                with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                    sequence_features, seq_midpred = detector(images)
                    del sequence_features
                    loss = criterion(seq_midpred, targets)
                scaler.scale(loss / group_size).backward()
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                prediction = torch.sigmoid(seq_midpred.detach()).gt(
                    args.threshold_eval
                )
                label = targets.gt(0)
                metric_counts[0] += torch.logical_and(
                    prediction, label
                ).sum(dtype=torch.int64)
                metric_counts[1] += torch.logical_or(
                    prediction, label
                ).sum(dtype=torch.int64)
                loss_stats[0] += loss.detach().to(torch.float64)
                loss_stats[1] += 1
            del images, targets, seq_midpred, loss, prediction, label
        all_reduce_sum(metric_counts, runtime)
        all_reduce_sum(loss_stats, runtime)
        train_loss = (loss_stats[0] / loss_stats[1].clamp_min(1)).item()
        train_iou = (
            metric_counts[0].to(torch.float64)
            / metric_counts[1].clamp_min(1).to(torch.float64)
        ).item()
        del metric_counts, loss_stats
        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy (IoU) of prediction: %f' % train_iou)
        if swanlab_run is not None:
            swanlab_module.log({
                'train/loss': train_loss,
                'train/iou': train_iou,
                'train/lr': lr,
            }, step=epoch + 1)

        latest_state = None
        if runtime.is_main:
            latest_state = {
                'epoch': epoch,
                'model_name': args.model,
                'class_avg_iou': best_iou,
                'model_state_dict': unwrap_model(detector).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'model_config': checkpoint_model_config,
            }
        latest_path = checkpoints_dir / 'latest_model.pth'
        pending_path = checkpoints_dir / (
            'pending_epoch_%03d.pth' % (epoch + 1)
        )
        if runtime.is_main and args.isolated_eval and not runtime.distributed:
            atomic_torch_save(latest_state, pending_path)
            log_string('Saved pending validation checkpoint at %s' % pending_path)

        torch.cuda.empty_cache()

        use_isolated_eval = bool(args.isolated_eval and not runtime.distributed)
        if args.isolated_eval and runtime.distributed and epoch == start_epoch:
            log_string(
                'DDP uses in-process rank-0 validation; isolated subprocess '
                'validation is only used for single-GPU training.'
            )
        if use_isolated_eval:
            detector = detector.cpu()
            criterion = criterion.cpu()
            move_optimizer_state(optimizer, 'cpu')
            torch.cuda.empty_cache()

            metrics_path = experiment_dir / (
                'validation_metrics_epoch_%03d.json' % (epoch + 1)
            )
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(BASE_DIR, 'validate_BRTD.py'),
                    '--checkpoint_path', str(pending_path),
                    '--metrics_path', str(metrics_path),
                    '--gpu', parse_visible_devices(args.gpu)[0],
                    '--seqlen', str(args.seqlen),
                    '--datapath', args.datapath,
                    '--dataset', args.dataset,
                    '--experiment_dir', str(experiment_dir),
                    '--threshold_eval', str(args.threshold_eval),
                    '--amp', str(args.eval_amp),
                ],
                check=True,
            )
            with metrics_path.open('r', encoding='utf-8') as file:
                validation_metrics = json.load(file)
            mIoU_mid = float(validation_metrics['iou'])
            log_string('Eval avg class IoU of prediction: %f' % mIoU_mid)

            is_best = mIoU_mid >= best_iou
            if is_best:
                best_iou = mIoU_mid
            latest_state['class_avg_iou'] = best_iou
            atomic_torch_save(latest_state, latest_path)
            if is_best:
                best_path = checkpoints_dir / 'best_model.pth'
                atomic_torch_save(latest_state, best_path)
                log_string('Saved best model at %s' % best_path)
            if (epoch + 1) % 5 == 0 or epoch + 1 == args.epoch:
                epoch_path = checkpoints_dir / (
                    'epoch_%d_model.pth' % (epoch + 1)
                )
                atomic_torch_save(latest_state, epoch_path)
            del validation_metrics, latest_state

            detector = detector.cuda()
            criterion = criterion.cuda()
            move_optimizer_state(optimizer, 'cuda')
            log_string('Best mIoU_mid: %f' % best_iou)
            if swanlab_run is not None:
                swanlab_module.log({
                    'eval/iou': mIoU_mid,
                    'eval/best_iou': best_iou,
                }, step=epoch + 1)
            continue

        distributed_barrier(runtime)
        if runtime.is_main:
            log_string('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
            eval_loss, mIoU_mid = evaluate_model(
                unwrap_model(detector),
                criterion,
                TEST_DATASET,
                args,
                runtime.device,
            )
            log_string('Eval mean loss: %f' % eval_loss)
            log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

            is_best = mIoU_mid >= best_iou
            if is_best:
                best_iou = mIoU_mid
            log_string('Best mIoU_mid: %f' % best_iou)
            latest_state['class_avg_iou'] = best_iou
            atomic_torch_save(latest_state, latest_path)
            if is_best:
                best_path = checkpoints_dir / 'best_model.pth'
                atomic_torch_save(latest_state, best_path)
                log_string('Saved best model at %s' % best_path)
            if (epoch + 1) % 5 == 0 or epoch + 1 == args.epoch:
                epoch_path = checkpoints_dir / (
                    'epoch_%d_model.pth' % (epoch + 1)
                )
                atomic_torch_save(latest_state, epoch_path)
            if swanlab_run is not None:
                swanlab_module.log({
                    'eval/loss': eval_loss,
                    'eval/iou': mIoU_mid,
                    'eval/best_iou': best_iou,
                }, step=epoch + 1)
            del latest_state

        best_iou = broadcast_object(
            best_iou if runtime.is_main else None,
            runtime,
        )
        distributed_barrier(runtime)

    if swanlab_run is not None and hasattr(swanlab_module, 'finish'):
        swanlab_module.finish()

    del detector, criterion, optimizer, trainDataLoader
    torch.cuda.empty_cache()
    distributed_barrier(runtime)
    is_main = runtime.is_main
    finalize_distributed(runtime)
    return is_main


if __name__ == '__main__':
    args = parse_args()
    if launch_with_torchrun_if_needed(
        __file__, args.gpu, args.gpu_num
    ):
        raise SystemExit(0)
    run_followup_test = main(args)

    if run_followup_test and args.run_test_after_train:
        subprocess.run(
            [
                sys.executable,
                os.path.join(BASE_DIR, 'test_BRTD.py'),
                '--gpu', parse_visible_devices(args.gpu)[0],
                '--seqlen', str(args.seqlen),
                '--datapath', args.datapath,
                '--dataset', args.dataset,
                '--log_dir', args.log_dir,
                '--logpath', args.savepath,
                '--amp', str(args.eval_amp),
                '--eval_chunk_rows', str(args.eval_chunk_rows),
            ],
            check=True,
        )
