import numpy as np

# Step 1: Generate a random input image (28x28x3) and kernels
input_image = np.random.rand(3, 28, 28)  # Random input image
num_kernels = 10  # Number of filters
kernel_size = (3, 5, 5)  # Height, Width, Depth (matches input channels)
padding = 0

# Create random kernels (filters)
kernels = np.random.rand(num_kernels, *kernel_size)  # Shape: (10, 5, 5, 3)

# Step 2: Extract patches from the input image
# Example batch input: (batch_size, height, width, channels)
batch_input_image = np.random.rand(32, 28, 28, 3)  # Shape: (32, 28, 28, 3) for 32 images

def extract_patches_batch(batch_images, kernel_height, kernel_width, stride=1, padding=0):
    batch_size, H, W, C = batch_images.shape
    if padding > 0:
        batch_images = np.pad(batch_images, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    
    out_height = (H - kernel_height) // stride + 1
    out_width = (W - kernel_width) // stride + 1
    
    all_patches = []
    
    for b in range(batch_size):
        image = batch_images[b]
        patches = []
        for i in range(0, H - kernel_height + 1, stride):
            for j in range(0, W - kernel_width + 1, stride):
                patch = image[i:i+kernel_height, j:j+kernel_width, :].flatten()
                patches.append(patch)
        all_patches.append(patches)
    
    return np.array(all_patches), out_height, out_width

# Extract patches for the batch
patches_batch, out_height, out_width = extract_patches_batch(batch_input_image, 5, 5, padding=padding)



flattened_kernels = kernels.reshape(num_kernels, -1)  # Shape: (10, 75)

# Reshape patches to (batch_size * num_patches, flattened_kernel_size)
num_patches = patches_batch.shape[1]  # Number of patches per image
flattened_patches = patches_batch.reshape(-1, flattened_kernels.shape[1])  # Shape: (batch_size * num_patches, 75)

# Perform matrix multiplication
output_batch = np.dot(flattened_patches, flattened_kernels.T)  # Shape: (batch_size * num_patches, 10)

# Reshape output to (batch_size, out_height, out_width, num_kernels)
output_feature_map_batch = output_batch.reshape(batch_input_image.shape[0], out_height, out_width, num_kernels)

# Print the output shape for the batch
print("Output feature map batch shape:", output_feature_map_batch.shape)

