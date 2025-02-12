import torch
from transformers import AutoModelForCausalLM
import os
from huggingface_hub import login
import time 
from sim import LatencySimulatorMode, patch_adamw_optimizer

patch_adamw_optimizer()

# Retrieve token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Authenticate
if hf_token:
    login(hf_token)
    print("Successfully logged into Hugging Face!")
else:
    print("HF_TOKEN environment variable not set.")

# Use Metal backend for Apple Silicon
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load a pre-trained model.
# (Replace model_name with an accessible model; here we assume a 1B model as an example)
model_name = "meta-llama/Llama-3.2-1B"  # Example; adjust if needed.
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.train()  # set model to training mode

# Generate synthetic tokenized inputs.
batch_size = 1
seq_length = 32
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
labels = input_ids.clone()  # use same tensor as labels

# Optimizer setup.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# For example, simulate on the MPS device if available.
sim_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Create an instance of LatencySimulatorMode from YAML.
# (Ensure that "gpu_latency_profile.yaml" exists in the current directory.)
sim_mode = LatencySimulatorMode.from_yaml("h100_profile.yaml", sim_device="cpu")

# Start time measurement for the overall training iteration.
start_time = time.perf_counter()

# Run one training iteration inside the simulator mode.
with sim_mode:
    outputs = model(input_ids=input_ids, labels=labels)  # forward pass (computes loss)
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()  # backward pass (computes gradients)
    optimizer.step()  # update parameters

end_time = time.perf_counter()
real_time = end_time - start_time

# Calculate simulated training speed (tokens processed per simulated second).
total_tokens = batch_size * seq_length
# Note: We use the total simulated delay accumulated by our simulator.
simulated_time = sim_mode.total_delay
tokens_per_sec = total_tokens / simulated_time if simulated_time > 0 else float('inf')

##########################
# Print Final Latencies  #
##########################
print("\n--- Final Latency Report ---")
print(f"Total simulated delay: {simulated_time:.6f} sec")
print(f"Real wall-clock time for training iteration: {real_time:.6f} sec")
print(f"Tokens processed: {total_tokens}")
print(f"Simulated training speed: {tokens_per_sec:.2f} tokens/sec")
print("\nPer-op latency breakdown:")
for op, delay in sim_mode.delay_breakdown.items():
    print(f"{op}: {delay:.6f} sec")
print("----------------------------")