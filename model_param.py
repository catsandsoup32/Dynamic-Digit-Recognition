import torch
import torch.nn as nn

from NEW_models import CNN_9, CNN_24

def print_num_parameters(model):
    # Count the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Print the total number of parameters
    print(f"Total number of parameters: {total_params}")

print_num_parameters(CNN_24())