import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import yaml

import torch.optim.adamw as adamw_module


def patch_adamw_optimizer():
    _original_single_tensor_adamw = adamw_module._single_tensor_adamw

    def patched_single_tensor_adamw(*args, **kwargs):
        try:
            return _original_single_tensor_adamw(*args, **kwargs)
        except ZeroDivisionError:
            # Instead of dividing by zero, substitute a small positive number
            return 1e-8

    adamw_module._single_tensor_adamw = patched_single_tensor_adamw

def convert_to_meta(obj):
    """Recursively convert any tensor in obj to a meta tensor."""
    if isinstance(obj, torch.Tensor):
        return obj.to("meta")
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_to_meta(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: convert_to_meta(v) for k, v in obj.items()}
    else:
        return obj

def create_dummy(meta_out, target_device, is_backward=False):
    if meta_out is None:
        return None
    if isinstance(meta_out, torch.Tensor):
        if is_backward:
            # Return a tensor filled with ones (nonzero) and with requires_grad=True.
            dummy = torch.ones(meta_out.shape, dtype=meta_out.dtype, device=target_device)
            dummy.requires_grad_()
            return dummy
        else:
            return torch.empty(meta_out.shape, dtype=meta_out.dtype, device=target_device)
    elif isinstance(meta_out, tuple):
        return tuple(create_dummy(x, target_device, is_backward) for x in meta_out)
    elif isinstance(meta_out, list):
        return [create_dummy(x, target_device, is_backward) for x in meta_out]
    else:
        return meta_out

def get_target_device(obj):
    """
    Recursively collect devices from any tensor(s) in obj.
    Returns the first found device (or CPU if none found).
    """
    devices = []
    def _collect(o):
        if isinstance(o, torch.Tensor):
            devices.append(o.device)
        elif isinstance(o, (list, tuple)):
            for x in o:
                _collect(x)
        elif isinstance(o, dict):
            for v in o.values():
                _collect(v)
    _collect(obj)
    return devices[0] if devices else torch.device("cpu")


class LatencySimulatorMode(TorchDispatchMode):
    def __init__(self, latency_profile, sim_device=None):
        """
        latency_profile: dictionary mapping op names to a dict with key "size_latency"
        sim_device: default device for dummy outputs (e.g. torch.device("mps"))
        """
        super().__init__()
        self.profile = latency_profile
        self.total_delay = 0.0
        self.delay_breakdown = {}
        self.sim_device = sim_device if sim_device is not None else torch.device("cpu")

    @classmethod
    def from_yaml(cls, yaml_file, sim_device=None):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        latency_profile = config.get("latency_profile", {})
        return cls(latency_profile, sim_device)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        op_name = f"{func.__module__}.{func.__name__}"

        # Special-case: bypass scalar extraction so that .item() works
        if "local_scalar_dense" in op_name:
            return func(*args, **kwargs)

        # Determine a representative tensor size for latency calculation.
        size = None
        # First, try to find the first tensor among args.
        for arg in args:
            if isinstance(arg, torch.Tensor):
                s = arg.numel()
                if not isinstance(s, int):
                    try:
                        size = int(s.item())
                    except Exception:
                        size = int(s)
                else:
                    size = s
                break

        # If no single tensor was found, handle the case when the first argument is a list or tuple.
        if size is None:
            if args and isinstance(args[0], (list, tuple)):
                first_elem = args[0]
                # If the first element of this list is a tensor, assume it's a list of tensors.
                if len(first_elem) > 0 and isinstance(first_elem[0], torch.Tensor):
                    size = sum(int(t.numel()) for t in first_elem if isinstance(t, torch.Tensor))
                else:
                    # Otherwise assume it is a list of dimensions.
                    shape = first_elem
                    size = 1
                    for dim in shape:
                        if isinstance(dim, torch.Tensor):
                            # If it's a one-element tensor, use that element;
                            # otherwise, default to 0.
                            if dim.numel() == 1:
                                size *= int(dim.item())
                            else:
                                size = 0
                                break
                        else:
                            size *= int(dim)
            else:
                size = 0

        # Now 'size' should be a plain Python int.
        delay = 0.0
        if op_name in self.profile:
            delay = self._get_latency(op_name, size)
        else:
            raise ValueError(f"no ops: {op_name}")
        self.total_delay += delay
        self.delay_breakdown[op_name] = self.delay_breakdown.get(op_name, 0.0) + delay

        if delay > 0:
            time.sleep(delay)

        # Convert arguments to meta tensors.
        meta_args = convert_to_meta(args)
        meta_kwargs = convert_to_meta(kwargs)
        try:
            meta_out = func(*meta_args, **meta_kwargs)
        except NotImplementedError:
            # Special-case for embedding op.
            if "embedding" in op_name and len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
                weight = args[0]
                input_tensor = args[1]
                embed_dim = weight.shape[-1]
                meta_shape = list(input_tensor.shape) + [embed_dim]
                meta_dtype = weight.dtype
                meta_out = torch.empty(meta_shape, dtype=meta_dtype, device="meta")
            else:
                base = None
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        base = arg
                        break
                if base is not None:
                    meta_shape = base.shape
                    meta_dtype = base.dtype
                else:
                    meta_shape = ()
                    meta_dtype = torch.float32
                meta_out = torch.empty(meta_shape, dtype=meta_dtype, device="meta")

        # Determine target device from input arguments.
        target_device = get_target_device(args)
        if target_device is None:
            target_device = self.sim_device

        # Mark as backward if op name contains "backward" (case-insensitive)
        is_backward = "backward" in op_name.lower()

        dummy = create_dummy(meta_out, target_device, is_backward=is_backward)
        return dummy

    def _get_latency(self, op_name, size):
        mapping = self.profile[op_name].get("size_latency", {})
        known_sizes = sorted(int(k) for k in mapping.keys())
        if not known_sizes:
            return 0.0
        if size in known_sizes:
            return mapping.get(str(size), mapping.get(size))
        if size <= known_sizes[0]:
            s1, s2 = known_sizes[0], known_sizes[min(1, len(known_sizes)-1)]
        elif size >= known_sizes[-1]:
            s1, s2 = known_sizes[-2] if len(known_sizes) >= 2 else known_sizes[0], known_sizes[-1]
        else:
            for i in range(len(known_sizes)-1):
                if known_sizes[i] <= size <= known_sizes[i+1]:
                    s1, s2 = known_sizes[i], known_sizes[i+1]
                    break
        lat1 = mapping.get(str(s1), mapping.get(s1))
        lat2 = mapping.get(str(s2), mapping.get(s2))
        if s1 == s2:
            return lat1
        interpolated = lat1 + (lat2 - lat1) * (size - s1) / (s2 - s1)
        return interpolated


# ==============================================================================
# Example usage in a training simulation.
# ==============================================================================

if __name__ == "__main__":
    # Choose simulation device (e.g., use MPS if available)
    sim_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # Load YAML latency profile (ensure "gpu_latency_profile.yaml" exists) and set sim_device.
    sim_mode = LatencySimulatorMode.from_yaml("gpu_latency_profile.yaml", sim_device=sim_device)
    
    # Dummy test: simulate an embedding op.
    vocab_size, embed_dim = 32000, 512
    batch_size, seq_length = 1, 32
    weight = torch.randn(vocab_size, embed_dim, device=sim_device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=sim_device)
    with sim_mode:
        embeds = torch._ops.aten.embedding.default(weight, input_ids)
    print("Dummy embedding output shape:", embeds.shape)
    
    # Dummy test: simulate an addition op.
    a = torch.randn(1000, device=sim_device)
    b = torch.randn(1000, device=sim_device)
    with sim_mode:
        c = a + b
    print("Dummy addition result shape:", c.shape)
    
    # Print simulated latency information.
    print("Total simulated delay: {:.6f} sec".format(sim_mode.total_delay))
    print("Per-op delay breakdown:")
    for op, d in sim_mode.delay_breakdown.items():
        print(f"{op}: {d:.6f} sec")
