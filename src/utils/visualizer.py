def visualize_rectangle(context_indices, target_indices):
    SIZE = 14
    # Create a 14x14 board initialized with '-'
    grid = [['-' for _ in range(SIZE)] for _ in range(SIZE)]
    
    # Mark the target positions with 'T'
    for idx in target_indices:
        row = idx // SIZE
        col = idx % SIZE
        grid[row][col] = 'T'
    
    # Mark the context positions with 'c'
    for idx in context_indices:
        row = idx // SIZE
        col = idx % SIZE
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
