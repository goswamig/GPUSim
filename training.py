import torch
from transformers import AutoModelForCausalLM
from torch.utils._python_dispatch import TorchDispatchMode

# 1. Select and load the pre-trained ~1B parameter model (Pythia-1B) and use MPS device
model_name = "meta-llama/Llama-3.2-1B"  # 1-billion-parameter Transformer model&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.train()  # set model to training mode

# 2. Generate synthetic tokenized inputs (random IDs in the model's vocab range)
batch_size = 1
seq_length = 32
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
# Use the same tensor as labels so the model computes a loss for this dummy input
labels = input_ids.clone()

# 3. Define a custom TorchDispatchMode to count operations
class OpCountingMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = {}  # dictionary to store counts of ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Get operation name (including module to avoid name collisions)
        op_name = f"{func.__module__}.{func.__name__}"
        # Increment count for this operation
        self.counts[op_name] = self.counts.get(op_name, 0) + 1
        # Execute the actual operation
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

# 4. Run a single forward and backward pass with the OpCountingMode enabled
counter = OpCountingMode()
with counter:
    outputs = model(input_ids, labels=labels)   # forward pass (computes loss)
    loss = outputs.loss
    loss.backward()                            # backward pass (computes gradients)

print(f"count of ops: {len(counter.counts)}")

# 5. After exiting the context, print the number of times each operation was used
for op, count in counter.counts.items():
    print(f"{op}: {count}")
