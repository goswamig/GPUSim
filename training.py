import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils._python_dispatch import TorchDispatchMode
from torch.profiler import profile, ProfilerActivity
import os
from huggingface_hub import login

# Retrieve token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Authenticate
if hf_token:
    login(hf_token)
    print("Successfully logged into Hugging Face!")
else:
    print("HF_TOKEN environment variable not set.")

# Use Metal backend for MacBook Air with M1/M2 chips
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load real LLaMA model from Hugging Face (replace as needed)
model_name = "meta-llama/Llama-3.2-1B"  # Example for a 1B model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.train()  # Set model to training mode

# Generate synthetic tokenized inputs (random IDs in the model's vocab range)
batch_size = 1
seq_length = 32
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
# Use the same tensor as labels so the model computes a loss for this dummy input
labels = input_ids.clone()

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Custom TorchDispatchMode to count operations
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

# Profile forward and backward passes with operation counts
counter = OpCountingMode()
with counter:
    outputs = model(input_ids=input_ids, labels=labels)  # Forward pass (computes loss)
    loss = outputs.loss

    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Backward pass (computes gradients)
    optimizer.step()  # Update model parameters

print(f"count of ops: {len(counter.counts)}")

# After exiting the context, print the number of times each operation was used
for op, count in counter.counts.items():
    print(f"{op}: {count}")
