from src.masks.custom_multiblock import MultiBlock

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

# Example usage:
multiblock = MultiBlock(
    block_scale=(0.15, 0.2),
    block_aspect_ratio=(0.75, 1.5)
)
context_indices, target_indices = multiblock(5)

print("Context block indices:", context_indices)
print("Target block indices:", target_indices)

# Visualize the rectangle
visualize_rectangle(context_indices[0].tolist(), target_indices[0].tolist())
print(f"Percent of context = {len(context_indices[0].tolist()) / 14**2 * 100}%")

def simulate(n=1000):
    percents = []
    for i in range(n):
        context_indices, target_indices = multiblock(5)
        percents.append(len(context_indices[0].tolist()) / 14**2 * 100)
    
    print(F"Avg: {sum(percents) / len(percents)}%")
    
    
simulate(10000)
# 50%
