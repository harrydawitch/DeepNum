import numpy as np
from Components.Layers.Base import Layer
from Components.Utils import *

class MaxPooling(Layer):
    def __init__(self,  filter_size=None, stride=1, padding=0):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.name = "maxpooling"


    def forward(self, inputs):
        self.data_in = inputs
        N, C, H, W = inputs.shape
        filter_h, filter_w = self.filter_size, self.filter_size

        inputs = inputs.reshape(N * C, 1 , H, W)

        out_h, out_w = get_output_pad(inputs, filter_h, filter_w, self.stride, self.padding)


        # im2col transformation
        self.X_col = im2col(inputs, filter_h, filter_w, self.stride, self.padding)
        self.max_indexes = np.argmax(self.X_col, axis= 0)
        

        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        return out
    

    def backward(self, dout):
        N, C, H, W = self.data_in.shape
        filter_h, filter_w = self.filter_size, self.filter_size
        dX_col = np.zeros_like(self.X_col)


        dout_reshaped = dout.transpose(2, 3, 0, 1).ravel()
        
        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_reshaped
        shape= (N * C, 1, H, W)
        dX = col2im(dX_col, shape, filter_h, filter_w, self.stride, self.padding)
        dX = dX.reshape(N, C, H, W)

        return dX




