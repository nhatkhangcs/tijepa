def visualize_rectangle(context_indices, target_indices, p=28):
    # Create a 14x14 board initialized with '-'
    grid = [['-' for _ in range(p)] for _ in range(p)]
    
    # Mark the target positions with 'T'
    for idx in target_indices:
        row = idx // p
        col = idx % p
        grid[row][col] = 'T'
    
    # Mark the context positions with 'c'
    for idx in context_indices:
        row = idx // p
        col = idx % p
        grid[row][col] = 'c'
    
    # Print the board
    for row in grid:
        print(' '.join(row))

def print_tensor_with_precision(tensor, precision=4):
  """Prints a tensor with the specified number of decimal places.

  Args:
    tensor: The tensor to print.
    precision: The number of decimal places to print.
  """

  # Convert the tensor to a numpy array for easier formatting
  tensor = tensor.tolist()

  # Print the tensor with the specified precision
  for element in tensor:
      print(f"{element:<8.{precision}f}", end=" ")
  print()

def print_sample_of_tensor(tensor):
    print(tensor[0, :5, :5])

from torch import nn

def print_linear_weights_and_biases(module, name=""):
    if not isinstance(module, nn.Module):
        return
    
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            # Print weights and biases of the Linear layer
            print(f"\n{name}.{child_name}")
            print(child_module.weight[:, :])
            
            if child_module.bias is not None:
                print(child_module.bias[:5])
        else:
            # Recursively search through child modules
            print_linear_weights_and_biases(child_module, f"{name}.{child_name}")

def print_transformer_weights(model):
    
    # Search through all transformer blocks
    for idx, block in enumerate(model.layers):
        print(f"\n=== Transformer Block {idx + 1} ===")
        print_linear_weights_and_biases(block, f"layers[{idx}]")
    
    # Print final output Linear layer weights and bias
    print("\nFinal Output Layer - Linear Layer Weights (first 5x5):")
    print(model.fc_out.weight[:5, :5])
    
    print("Final Output Layer - Linear Layer Bias (first 5):")
    print(model.fc_out.bias[:5])
