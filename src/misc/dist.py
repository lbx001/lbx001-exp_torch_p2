"""Distributed training utilities."""
import os
import pickle
import torch
import torch.distributed as dist

__all__ = [
    'is_dist_available_and_initialized', 'get_rank', 'get_world_size',
    'is_main_process', 'save_on_master', 'all_gather', 'reduce_dict',
    'setup_for_distributed',
]


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_gather(data):
    """Gather arbitrary picklable data from all processes."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(s.item()) for s in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size.item() != max_size:
        padding = torch.empty(max_size - local_size, dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    result = []
    for size, t in zip(size_list, tensor_list):
        buffer = t[:size].cpu().numpy().tobytes()
        result.append(pickle.loads(buffer))
    return result


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dict values across all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys])
        dist.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(keys, values)}


def setup_for_distributed(is_master: bool):
    """Disable printing on non-master processes."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
