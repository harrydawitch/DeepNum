import numpy as np
from layer_utils import im2col, col2im

class Conv2D:
    def __init__(self, n_filters= None, filter_size= None, stride= 1, padding= 0, activation= None, regularizer= None):
        self.n_filters= n_filters
        self.stride = stride
        self.padding = padding 
        self.activation = activation
        self.regularizer = regularizer

        self.filter_size = filter_size

        

    def forward(self, input_data, filters, stride=1, pad=0):
        """
        Perform 2D convolution using im2col for NHWC format.
        Args:
            input_data: Input data, shape (N, H, W, C)
            filters: Convolution filters, shape (filter_h, filter_w, in_channels, out_channels)
            bias: Bias term, shape (out_channels,)
            stride: Stride of convolution
            pad: Padding size
        Returns:
            out: Output data, shape (N, out_h, out_w, out_channels)
        """

        N, H, W, C = input_data.shape
        filter_h, filter_w, in_channels, out_channels = filters.shape

        assert C == in_channels, "Input channels must match filter channels."

        # Output dimensions
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        # im2col transformation
        col = im2col(input_data, filter_h, filter_w, stride, pad)

        # Reshape filters to match im2col format
        col_filters = filters.reshape(-1, out_channels)

        # Perform matrix multiplication
        out = np.dot(col, col_filters) 

        # Reshape to NHWC output
        out = out.reshape(N, out_h, out_w, out_channels)
        return out






