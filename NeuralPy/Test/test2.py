import numpy as np

np.random.seed(32)
# Step 1: Generate a random input image (28x28x3) and kernels
input_image = np.random.randint(1, 9, (4, 4))  # Random input image
num_kernels = 10  # Number of filters
kernel_size =  np.random.randn(2, 2) ** 2   # Height, Width, Depth (matches input channels)

print('Input image')
print(input_image)
print()
print('Kernel')
print(kernel_size)


def conv2d(image, kernel):
    input_h, input_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = (input_h - kernel_h) + 1
    output_w = (input_w - kernel_w) + 1

    output = np.zeros((output_h, output_w))

    for height in range(output_h):
        for width in range(output_w):
            patch = image[height:height+kernel_h, width:width+kernel_w]
            output[height, width] = np.sum(np.multiply(patch, kernel))
    
    return output

print()
print('Output')
print(conv2d(input_image, kernel_size))