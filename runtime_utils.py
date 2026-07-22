"""Runtime helpers shared by training and evaluation entry points."""

import os
import subprocess
import sys
import tempfile
import datetime
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def parse_visible_devices(gpu):
    devices = [item.strip() for item in str(gpu).split(',') if item.strip()]
    if not devices:
        raise ValueError('--gpu must name at least one CUDA device.')
    if len(set(devices)) != len(devices):
        raise ValueError('--gpu contains duplicate CUDA devices: %s' % gpu)
    return devices


def launch_with_torchrun_if_needed(script_path, gpu, gpu_num, argv=None):
    """Relaunch a multi-GPU command through torchrun without a shell."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size > 1 or gpu_num <= 1:
        return False

    devices = parse_visible_devices(gpu)
    if len(devices) != gpu_num:
        raise ValueError(
            '--gpu_num=%d, but --gpu exposes %d device(s): %s'
            % (gpu_num, len(devices), gpu)
        )
    if argv is None:
        argv = sys.argv[1:]

    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node',
        str(gpu_num),
        str(Path(script_path).resolve()),
    ] + list(argv)
    subprocess.run(command, check=True, env=environment)
    return True


@dataclass(frozen=True)
class DistributedRuntime:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def initialize_distributed(gpu, gpu_num):
    devices = parse_visible_devices(gpu)
    if len(devices) != gpu_num:
        raise ValueError(
            '--gpu_num=%d, but --gpu exposes %d device(s): %s'
            % (gpu_num, len(devices), gpu)
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        if world_size != gpu_num:
            raise ValueError(
                'torchrun WORLD_SIZE=%d does not match --gpu_num=%d.'
                % (world_size, gpu_num)
            )
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(hours=24),
        )
    else:
        if gpu_num != 1:
            raise RuntimeError('Multi-GPU training must be launched through torchrun.')
        local_rank = 0
        rank = 0
        torch.cuda.set_device(local_rank)

    return DistributedRuntime(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device('cuda', local_rank),
    )


def distributed_barrier(runtime):
    if runtime.distributed:
        dist.barrier()


def broadcast_object(value, runtime, source=0):
    if not runtime.distributed:
        return value
    values = [value if runtime.rank == source else None]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def all_reduce_sum(tensor, runtime):
    if runtime.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def load_checkpoint(path, map_location='cpu'):
    """Load tensor/state-dict checkpoints without executing arbitrary pickle."""
    return torch.load(path, map_location=map_location, weights_only=True)


def atomic_torch_save(state, destination):
    """Atomically replace a checkpoint after a complete same-filesystem write."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix='.%s.' % destination.name,
            suffix='.tmp',
            dir=str(destination.parent),
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(state, temporary_path)
        os.replace(str(temporary_path), str(destination))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def finalize_distributed(runtime):
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
