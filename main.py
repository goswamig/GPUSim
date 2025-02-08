import torch
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

class OpCountingMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.op_counts = defaultdict(int)  # Dictionary to store op counts

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        op_name = f"{func.__module__}.{func.__name__}"  # e.g., "torch._ops.aten.add.Tensor"

        # Increment call count for this operation
        self.op_counts[op_name] += 1

        # Forward to the actual implementation of the op
        return func(*args, **kwargs)

    def report(self):
        """Prints the logged operations and their counts."""
        print("\n[Op Counting Report]")
        for op, count in sorted(self.op_counts.items()):
            print(f"{op}: {count} times")
        print("[End of Report]\n")

# Example usage:
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.ones(3)

    with OpCountingMode() as mode:
        z = x + y  # Should count an "aten.add" op
        w = z.sum()  # Should count an "aten.sum" op
        w_item = w.item()  # Conversion to Python number, counts another op

    # Print results
    mode.report()
