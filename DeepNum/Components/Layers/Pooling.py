import numpy as np
from Components.Layers.Base import Layer
from Components.Utilities.Utils import im2col, col2im

class Pooling(Layer):
    def __init__(self, mode=None, filter_size=None, stride=1, padding=0):
        super().__init__()
        self.mode = mode
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.name = "Pooling"


    def forward(self, inputs):
        self.data_in = inputs
        N, H, W, C = self.data_in.shape
        filter_h, filter_w = self.filter_size

        # Output dimensions
        out_h = (H + 2 * self.padding - filter_h) // self.stride + 1
        out_w = (W + 2 * self.padding - filter_w) // self.stride + 1

        # im2col transformation
        self.X_col = im2col(self.data_in, filter_h, filter_w, self.stride, self.padding)
        col_reshape = self.X_col.reshape(N * out_h * out_w, filter_h * filter_w, C)


        if self.mode == 'max':
            max_values = np.max(col_reshape, axis=1, keepdims=True)
            one_hot_mask = (col_reshape == max_values).astype(np.float32)
            pooling = np.sum(one_hot_mask * col_reshape, axis=1)

        elif self.mode == 'average':
            pooling = np.average(col_reshape, axis=1)


        # Reshape to NHWC output
        pooling = pooling.reshape(N, out_h, out_w, C)

        return pooling

    def backward(self, dout):
        N, H, W, C = self.data_in.shape
        filter_h, filter_w = self.filter_size

        # Output dimensions
        out_h = (H + 2 * self.padding - filter_h) // self.stride + 1
        out_w = (W + 2 * self.padding - filter_w) // self.stride + 1

        # Reshape dout to match the shape of the im2col output
        dout_reshaped = dout.reshape(-1, C)

        # Reshape im2col output for pooling patches
        col_reshape = self.X_col.reshape(N * out_h * out_w, filter_h * filter_w, C)

        if self.mode == 'max':
            max_values = np.max(col_reshape, axis=1, keepdims=True)
            one_hot_mask = (col_reshape == max_values).astype(np.float32)
            dX_col = one_hot_mask * dout_reshaped[:, None]
            dX_col = dX_col.reshape(self.X_col.shape)

        elif self.mode == 'average':
            dX_col = np.repeat(dout_reshaped[:, None], filter_h * filter_w, axis=1) / (filter_h * filter_w)
            dX_col = dX_col.reshape(self.X_col.shape)

        # Convert the gradient in column format back to the original input shape
        dX = col2im(dX_col, (N, H, W, C), filter_h, filter_w, self.stride, self.padding)

        return dX


