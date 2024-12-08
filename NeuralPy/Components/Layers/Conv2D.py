import numpy as np
from ..Utils import pick_initializer, activate, im2col, col2im

class Conv2D:
    def __init__(self, n_filters= None, filter_size= None, stride= 1, padding= 0, activation= None, regularizer= None, initializer= None):
        self.output= n_filters
        self.input = None
        self.stride = stride
        self.padding = padding 
        self.activation = activation
        self.regularizer = regularizer
        self.initializer = initializer
        self.filter_size = filter_size
        self.learnable = True
        self.name = 'Conv2D'

        if activation is not None:
            self.activation = activate(name= activation)



    def init_params(self):

        if isinstance(self.filter_size, (tuple, list)) and len(self.filter_size) == 2:
            height, width = self.filter_size
        
            
            if self.initializer:
                self.weight = pick_initializer(initializers= self.initializer, 
                                               shape= (height, width, self.input, self.output), 
                                               name= self.name)
                

            else:
                self.weight = np.random.randn(height, width, self.input, self.output)

            self.filter_height = height
            self.filter_width= width

        self.bias = np.zeros(self.output)

        assert len(self.filter_size) == 2, "Filter size must be a tuple/list contain 2 values -> (Height, Width)"


        
    def forward(self, inputs):
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

        self.data_in = inputs
        N, H, W, C = self.data_in.shape
        filter_h, filter_w, in_channels, out_channels = self.weight.shape

        assert C == in_channels, "Input channels must match filter channels."

        # Output dimensions
        out_h = (H + 2 * self.padding - filter_h) // self.stride + 1
        out_w = (W + 2 * self.padding - filter_w) // self.stride + 1

        # im2col transformation
        self.X_col = im2col(self.data_in, filter_h, filter_w, self.stride, self.padding)

        # Reshape filters to match im2col format
        self.W_col = self.weight.reshape(-1, out_channels)

        # Perform matrix multiplication
        out = np.dot(self.X_col, self.W_col) + self.bias

        # Reshape to NHWC output
        self.data_out = out.reshape(N, out_h, out_w, out_channels)

        if self.activation is not None:
            self.data_out= self.activation.forward(Z= self.data_out)
        
        return self.data_out



    def backward(self, dout):
        N, h_out, w_out, c_out = dout.shape
        filter_height, filter_width, in_channels, _ = self.weight.shape
        
        self.db = np.sum(dout, axis=(0,1,2)) 

        dout=  dout.reshape(N*h_out*w_out, c_out)

        dX_col = np.dot(dout, self.W_col.T)
        dw_col = np.dot(self.X_col.T, dout)

        dX = col2im(dX_col, self.data_in.shape, filter_height, filter_width, self.stride, self.padding)
        self.dw = dw_col.reshape(filter_height, filter_width, in_channels, c_out)

        return dX
