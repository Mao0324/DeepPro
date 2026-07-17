"""
Author: Benny
Date: Nov 2019
"""
import argparse
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clean_state_dict(state_dict):
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


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
    parser.add_argument('--use_swanlab', type=int, default=1, choices=[0, 1], help='Use SwanLab logging [default: 1]')
    parser.add_argument('--swanlab_project', type=str, default='DeepPro', help='SwanLab project name')
    parser.add_argument('--seed', type=int, default=46, help='Random seed')
    parser.add_argument('--base_ckpt', type=str, default='', help='DeepPro-Plus checkpoint used to initialize BRTD')
    parser.add_argument('--base_lr_mult', type=float, default=0.1, help='Learning-rate multiplier for pretrained layers')
    parser.add_argument('--brtd_use_background', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_adaptive_tdc', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_use_gate', type=int, default=1, choices=[0, 1])
    parser.add_argument('--brtd_zero_init', type=int, default=1, choices=[0, 1])
    parser.add_argument('--run_test_after_train', type=int, default=1, choices=[0, 1])
    parser.add_argument("--spatial_ckpt", type=str, default="")
    parser.add_argument("--st_ckpt", type=str, default="")
    parser.add_argument("--freeze_pretrained", type=int, default=1)

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if args.gpu_num == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

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
    TRAIN_DATASET = TrainIRSeqDataLoader(
        args.dataset,
        data_root=root,
        seq_len=SEQ_LEN,
        sample_rate=args.sample_rate,
        patch_size=args.patch_size,
        transform=None,
    )
    print("start loading test data ...")
    TEST_DATASET = TestIRSeqDataLoader(
        args.dataset,
        data_root=root,
        seq_len=SEQ_LEN,
        cat_len=int(SEQ_LEN * 0.1),
        transform=None,
    )

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d sequences" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('networks/models/%s.py' % args.model, str(experiment_dir))
    checkpoint_model_config = {}

    if "BRTD" in args.model:
        checkpoint_model_config = {
            'use_background': bool(args.brtd_use_background),
            'adaptive_tdc': bool(args.brtd_adaptive_tdc),
            'use_gate': bool(args.brtd_use_gate),
            'zero_init': bool(args.brtd_zero_init),
        }
        shutil.copy('networks/layers/brtd_adapter.py', str(experiment_dir))
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
        detector = MODEL.detector(NUM_CLASSES, SEQ_LEN, SEQ_LEN)

    if args.base_ckpt:
        checkpoint = torch.load(args.base_ckpt, map_location='cpu')
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

    if args.gpu_num > 1:
        detector = torch.nn.DataParallel(detector)#, device_ids=list(np.arange(args.gpu_num)))
    detector = detector.cuda()
    # criterion = MODEL.bceloss().cuda()
    # criterion = MODEL.HAMloss().cuda()
    criterion = MODEL.SoftLoUloss().cuda()

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

    best_iou = 0
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


    LEARNING_RATE_CLIP = 1e-5
    global_epoch = 0
    ## train
    for epoch in range(start_epoch, args.epoch):
        '''Train'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        num_batches = len(trainDataLoader)
        total_intersection_mid = 0
        total_union_mid = 0
        loss_sum = 0
        detector.train()

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
            bar_format=(
                '{desc}: {percentage:3.0f}%|{bar:30}| '
                '{n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}]'
            ),
        )

        for i, (images, targets) in train_bar:
            optimizer.zero_grad()
            #torch.autograd.set_detect_anomaly = True
            images, targets = images.float().cuda(), targets.float().cuda()

            _, seq_midpred = detector(images)

            loss = criterion(seq_midpred, targets)
            loss.backward()
            optimizer.step()

            seq_midpred = torch.sigmoid(seq_midpred)
            midpred_choice = (seq_midpred.cpu().data.numpy() > args.threshold_eval) * 1.
            batch_label    = targets.cpu().data.numpy()
            total_intersection_mid += np.sum(midpred_choice * batch_label)
            total_union_mid += ((midpred_choice + batch_label)>0).astype(np.float32).sum()
            loss_sum += loss.item()
            train_bar.set_postfix(
                loss='%.4f' % loss.item(),
                lr='%.2e' % optimizer.param_groups[-1]['lr'],
                refresh=False,
            )
            # break
        train_loss = loss_sum / num_batches
        train_iou = total_intersection_mid / total_union_mid
        log_string('Training mean loss: %f' % train_loss)
        log_string('Training accuracy (IoU) of prediction: %f' % train_iou)
        if swanlab_run is not None:
            swanlab.log({
                'train/loss': train_loss,
                'train/iou': train_iou,
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
                    'model_config': checkpoint_model_config,
                }
            else:
                state = {
                    'epoch': epoch,
                    'model_state_dict': detector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': checkpoint_model_config,
                }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate'''
        with torch.no_grad():
            num_batches = 0
            total_intersection_mid = 0
            total_union_mid = 0
            loss_g_sum = 0
            detector = detector.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            # for i, (images, targets) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            #for seq_idx, seq_dataset in tqdm(enumerate(TEST_DATASET), total=len(TEST_DATASET), smoothing=0.9):
            eval_bar = tqdm(
                enumerate(TEST_DATASET),
                total=len(TEST_DATASET),
                desc='Eval  %03d/%03d' % (epoch + 1, args.epoch),
                smoothing=0.9,
                ascii=True,
                dynamic_ncols=False,
                ncols=100,
                mininterval=0.5,
                leave=True,
                file=sys.stdout,
                bar_format=(
                    '{desc}: {percentage:3.0f}%|{bar:30}| '
                    '{n_fmt}/{total_fmt} '
                    '[{elapsed}<{remaining}, {rate_fmt}]'
                ),
            )

            for seq_idx, seq_dataset in eval_bar:
                # if seq_idx % 3 > 0:
                #     continue
                seq_dataloader = torch.utils.data.DataLoader(seq_dataset, batch_size=1, shuffle=False)
                num_batches += len(seq_dataloader)
                for i, (images, targets, _, first_end) in enumerate(seq_dataloader):
                    images, targets = images.float().cuda(), targets.float().cuda()

                    _, seq_midpred = detector(images)
                    if seq_midpred.shape[-1] != targets.shape[-1]:
                        seq_midpred = F.interpolate(seq_midpred, size=targets.shape[-2:])

                    loss_g_sum += criterion(seq_midpred, targets).item()

                    seq_midpred = torch.sigmoid(seq_midpred)
                    pred_choice_mid = (seq_midpred.cpu().data.numpy() > args.threshold_eval) * 1.
                    batch_label     = targets.cpu().data.numpy()
                    total_intersection_mid += np.sum(pred_choice_mid * batch_label)
                    total_union_mid += ((pred_choice_mid + batch_label) > 0).astype(np.float32).sum()

            mIoU_mid = total_intersection_mid / total_union_mid
            eval_loss = loss_g_sum / float(num_batches)
            log_string('Eval mean loss: %f' % eval_loss)
            log_string('Eval avg class IoU of prediction: %f' % (mIoU_mid))

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
                        'model_config': checkpoint_model_config,
                    }
                else:
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU_mid,
                        'model_state_dict': detector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_config': checkpoint_model_config,
                    }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU_mid: %f' % best_iou)
            if swanlab_run is not None:
                swanlab.log({
                    'eval/loss': eval_loss,
                    'eval/iou': mIoU_mid,
                    'eval/best_iou': best_iou,
                }, step=epoch + 1)

        global_epoch += 1

    if swanlab_run is not None and hasattr(swanlab, 'finish'):
        swanlab.finish()


def path_remake(path):
    return path.replace(' ', r'\ ').replace('(', r'\(').replace(')', r'\)').replace('&', r'\&')


if __name__ == '__main__':
    args = parse_args()
    main(args)

    if args.run_test_after_train:
        os.system('python test.py --gpu %s --seqlen %d --datapath %s --dataset %s --log_dir %s' % (
                args.gpu, args.seqlen, path_remake(args.datapath), path_remake(args.dataset), path_remake(args.log_dir)))

