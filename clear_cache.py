import torch

# Clear GPU cache
torch.cuda.empty_cache()
print("GPU cache cleared.")

# Optionally, reset all allocated memory
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
print("GPU memory stats reset.")

# Print current memory usage
if torch.cuda.is_available():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("No GPU available.")