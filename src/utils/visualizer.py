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
