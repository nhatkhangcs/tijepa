import random
import torch

class MultiBlock:
    def __init__(self, block_scale=(0.15, 0.2), n_block=4, block_aspect_ratio=(0.75, 1.5), context_scale=(0.85, 1.0), device='cuda'):
        self.block_scale = block_scale  # Portion of the image area the block should cover
        self.n_block = n_block
        self.block_aspect_ratio = block_aspect_ratio
        self.context_scale = context_scale  # Portion of the image area for context
        self.grid_size = 14  # 14x14 patches for a 224x224 image
        self.total_patches = self.grid_size * self.grid_size  # 196 patches
        self.device = device

    def _sample_block(self, scale_range, aspect_ratio_range):
        """Sample a random block (w, h) based on the area and aspect ratio ranges."""
        # Step 1: Sample the area of the block as a percentage of the total image area
        area_percentage = random.uniform(*scale_range)
        block_area = area_percentage * self.total_patches  # Total number of patches covered by the block
        
        # Step 2: Sample the aspect ratio
        aspect_ratio = random.uniform(*aspect_ratio_range)
        
        # Step 3: Calculate width and height from the area and aspect ratio
        block_height = int(round((block_area / aspect_ratio) ** 0.5))  # h = sqrt(area / aspect_ratio)
        block_width = int(round(block_height * aspect_ratio))          # w = h * aspect_ratio

        # Step 4: Ensure the block stays within bounds of the grid
        block_width = min(block_width, self.grid_size)
        block_height = min(block_height, self.grid_size)
        
        return block_width, block_height

    def _get_block_indices(self, block_width, block_height):
        """Generate the indices for a block with the given width and height."""
        max_x_start = self.grid_size - block_width
        max_y_start = self.grid_size - block_height
        
        start_x = random.randint(0, max_x_start)
        start_y = random.randint(0, max_y_start)
        
        # Get the indices forming the block
        indices = []
        for y in range(start_y, start_y + block_height):
            for x in range(start_x, start_x + block_width):
                indices.append(y * self.grid_size + x)  # Convert 2D index to 1D
        
        return indices

    def __call__(self, batch_size):
        # Step 1: Randomly sample target blocks (shared across the batch)
        target_indices = set()
        for _ in range(self.n_block):
            block_width, block_height = self._sample_block(self.block_scale, self.block_aspect_ratio)
            block_indices = self._get_block_indices(block_width, block_height)
            target_indices.update(block_indices)  # Add the block indices to the target set
        
        # Convert target_indices to a tensor
        target_indices_tensor = torch.tensor(list(target_indices), dtype=torch.int64)

        # Step 2: Sample a large context block (shared across the batch)
        context_width, context_height = self._sample_block(self.context_scale, self.context_scale)
        context_indices = set(self._get_block_indices(context_width, context_height))

        # Remove overlap between context and target blocks
        context_indices = context_indices - target_indices
        
        # Convert context_indices to a tensor
        context_indices_tensor = torch.tensor(list(context_indices), dtype=torch.int64)

        # Step 3: Repeat the tensors across the batch size
        context_batch = context_indices_tensor.unsqueeze(0).repeat(batch_size, 1).to(self.device)  # (batch_size, num_context_indices)
        target_batch = target_indices_tensor.unsqueeze(0).repeat(batch_size, 1).to(self.device)    # (batch_size, num_target_indices)

        return context_batch, target_batch
