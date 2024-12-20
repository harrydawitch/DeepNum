import numpy as np
from Components.Utils import  *
from Components.Layers.Base import Layer


class Conv2D(Layer):
    def __init__(self, n_filters= None, filter_size= None, stride= 1, padding= 0, regularizer= None, initializer= 'glorot_uniform'):
        super().__init__()
        self.output= n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding 
        self.regularizer = regularizer
        self.initializer = initializer
        self.learnable= True
        self.parameters = {}
        self.grads = {}
        self.name = 'Conv2D'
        self.filter_size = filter_size

    def forward(self, X):
        '''
        X shape should be (N, C, H, W)
        '''
        if self.shape is None:
            self.shape = X.shape[find_shape(X, mode= 'features')]
            super().init_params()

        self.data_in = X

        N, _, H, W = self.data_in.shape
        filter_h, filter_w = self.filter_size, self.filter_size
        n_filters = self.parameters['Weight'].shape[0]

        out_h, out_w = get_output_pad(X, filter_h, filter_w, self.stride, self.padding)


        self.X_col = im2col(self.data_in, filter_h,  filter_w, self.stride, self.padding)
        W_row = self.parameters['Weight'].reshape(n_filters, -1)

        out = W_row @ self.X_col + np.expand_dims(self.parameters['bias'], axis= -1)
        out = out.reshape(n_filters, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward(self, dout):
        n_filters, _, filter_h, filter_w = self.parameters['Weight'].shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        shape = self.data_in.shape
        
        self.grads['db'] = np.sum(dout, axis=(0, 2, 3)).reshape(n_filters,)

        self.grads['dW'] = dout_reshaped @ self.X_col.T
        self.grads['dW'] = self.grads['dW'].reshape(self.parameters['Weight'].shape)

        W_flat = self.parameters['Weight'].reshape(n_filters, -1)

        dX_col = W_flat.T @ dout_reshaped

        dout = col2im(dX_col, shape, filter_h, filter_w, self.stride, self.padding)
        
        return dout

 
