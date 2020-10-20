# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager
#from torch.utils.checkpoint import detach_variable

# DeepSpeed Integration
import torch.distributed as dist
PARTITION_ACTIVATIONS = False
PA_CORRECTNESS_TEST= False

def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", torch.cuda.memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Max Memory Allocated ", torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Cache Allocated ", torch.cuda.memory_cached()/(1024*1024*1024), "GigaBytes")
        print("Max cache Allocated ", torch.cuda.max_memory_cached()/(1024*1024*1024), "GigaBytes")
        print(" ")
        #input("Press Any Key To Continue ..")

from megatron import get_args
from megatron.memory import allocate_mem_buff

from .initialize import get_data_parallel_rank
from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size

# DeepSpeed Integration
mp_rank = None #get_model_parallel_rank()
mp_size = None #get_model_parallel_world_size()
mp_group = None #get_model_parallel_group()


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


# Whether apply model parallelsim to checkpointed hidden states.
_CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = None

# DeepSpeed Integration
transport_stream = None
cuda_device = None


def detach_variable(inputs, device=None):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            requires_grad = inp.requires_grad

            if device is not None:
                x = inp.to(device=device)
            else:
                x = inp

            x = x.detach()
            x.requires_grad = requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def init_checkpointed_activations_memory_buffer():
    """Initializ the memory buffer for the checkpointed activations."""
    args = get_args()

    per_layer = args.batch_size * args.max_position_embeddings * \
                args.hidden_size // args.model_parallel_size
    assert args.num_layers % args.checkpoint_num_layers == 0, \
        'number of layers is not divisible by checkpoint-num-layers'
    num_checkpointer_layers = args.num_layers // args.checkpoint_num_layers
    numel = per_layer * num_checkpointer_layers
    dtype = torch.half
    if not args.fp16:
        dtype = torch.float
        
    global _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER
    assert _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is None, \
        'checkpointed activations memory buffer is already allocated.'
    _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER = allocate_mem_buff(
        'checkpointed activations', numel, dtype, track_usage=False)


def reset_checkpointed_activations_memory_buffer():
    """Reset the memory used for checkpointing."""
    if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
        _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.reset()


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def split_tensor_into_1d_equal_chunks(tensor):
    """Break a tensor into equal 1D chunks."""
    data = tensor.view(-1)
    partition_size = torch.numel(data) // get_model_parallel_world_size()
    start_index = partition_size * get_model_parallel_rank()
    end_index = start_index + partition_size
    return data[start_index:end_index]


def gather_split_1d_tensor(tensor):
    """Opposite of above function, gather values from model parallel ranks."""
    world_size = get_model_parallel_world_size()
    numel = torch.numel(tensor)
    numel_gathered = world_size * numel
    gathered = torch.empty(numel_gathered, dtype=tensor.dtype,
                           device=torch.cuda.current_device(),
                           requires_grad=False)
    chunks = [gathered[i*numel:(i+1)*numel] for i in range(world_size)]
    torch.distributed.all_gather(chunks, tensor,
                                 group=get_model_parallel_group())
    return gathered


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    model_parallel_seed = offset + get_model_parallel_rank()
    # Data parallel gets the original sedd.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, '
              'model parallel rank {}, and data parallel rank {} with '
              'model parallel seed: {} and data parallel seed: {}'.format(
                  torch.distributed.get_rank(), get_model_parallel_rank(),
                  get_data_parallel_rank(), model_parallel_seed,
                  data_parallel_seed), flush=True)
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                model_parallel_seed)


# DeepSpeed Integration
def get_partition_start(item):
    global mp_rank, mp_size, mp_group
    partition_size = get_partition_size(item)
    start = partition_size * mp_rank
    return int(start)


def get_partition_size(item):
    global mp_rank, mp_size, mp_group
    size = item.numel()
    partition_size = size / mp_size
    return int(partition_size)


def get_full_inputs(tensors):
    inputs = []
    for i in range(int(len(tensors) / 2) - 1):
        item = tensors[2 * i]
        size = tensors[2 * i + 1]
        partition_size = item.numel()
        tensor_size = partition_size * mp_size
        flat_tensor = torch.zeros([tensor_size], dtype=item.dtype, device=item.device)
        partitions = []
        for i in range(mp_size):
            part_i = flat_tensor.narrow(0, partition_size * i, partition_size)
            if i == mp_rank:
                part_i.copy_(item)
            partitions.append(part_i)
        dist.all_gather(partitions, partitions[mp_rank], group=mp_group)
        input_tensor = flat_tensor.view(list(size.numpy()))
        item.data = input_tensor.data

        inputs.append(item)
    inputs.append(tensors[-2])

    return tuple(inputs)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """
    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        # DeepSpeed Integration
        global mp_rank, mp_size, mp_group
        if mp_rank is None:
            mp_rank = get_model_parallel_rank()
            mp_size = get_model_parallel_world_size()
            mp_group = get_model_parallel_group()

        global cuda_device, transport_stream, PARTITION_ACTIVATIONS
        if cuda_device is None:
            if dist.get_rank() == 0:
                print(f"Partition Activations {PARTITION_ACTIVATIONS} and Correctness Check {PA_CORRECTNESS_TEST}")

            cuda_device = torch.cuda.current_device()
            # The transport stream is used to overlap the allgather communication for the activations
            # with the computation in the backward pass
            transport_stream = torch.cuda.Stream(device=cuda_device)

        if PARTITION_ACTIVATIONS:
            inputs = [item.detach().contiguous().view(-1).narrow(0, get_partition_start(item),
                                                                 get_partition_size(item)).clone() for item in
                      args[:-1]]
            inputs.append(args[-1])

        # just in case something funky is happening such as reuse of inputs
        inputs_cuda = [item.to(cuda_device) for item in args]

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            # DeepSpeed Integration
            outputs = run_function(*inputs_cuda)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            ctx.input_0_shape = args[0].data.shape
            args[0].data = split_tensor_into_1d_equal_chunks(args[0].data)
            args[0].data = _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER.add(
                args[0].data)
            
        # Store everything.
        # ctx.save_for_backward(*args)

        # DeepSpeed Integration
        del inputs_cuda

        if PARTITION_ACTIVATIONS:
            new_args = []
            for arg, inp in zip(args, inputs):
                size = torch.tensor(arg.size())
                arg.data = inp.data
                new_args.append(arg)
                new_args.append(size)
            ctx.save_for_backward(*new_args)
        else:
            ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), "
                               "please use .backward() if possible")
        # DeepSpeed Integration
        global cuda_device, transport_stream, PARTITION_ACTIVATIONS

        if PARTITION_ACTIVATIONS:
            with torch.cuda.stream(transport_stream):
                inputs = get_full_inputs(ctx.saved_tensors)
                detached_inputs = detach_variable(inputs)
        else:
            inputs = ctx.saved_tensors
            detached_inputs = detach_variable(inputs)

        if _CHECKPOINTED_ACTIVATIONS_MEMORY_BUFFER is not None:
            inputs[0].data = gather_split_1d_tensor(inputs[0].data)
            inputs[0].data = inputs[0].data.view(ctx.input_0_shape)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # DeepSpeed Integration
        if PARTITION_ACTIVATIONS:
            current_stream=torch.cuda.current_stream()
            current_stream.wait_stream(transport_stream)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None,) + grads


def checkpoint(function, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint."""
    return CheckpointFunction.apply(function, *args)


# DeepSpeed Integration
def partition_activations_in_checkpoint(partition_activation):
    global PARTITION_ACTIVATIONS
    PARTITION_ACTIVATIONS=partition_activation
    if dist.get_rank()  == 0:
        print(f"**************Partition Activations {PARTITION_ACTIVATIONS}************")
