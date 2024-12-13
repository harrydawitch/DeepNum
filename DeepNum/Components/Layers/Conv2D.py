import numpy as np
from Components.Utilities.Utils import  find_shape
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



    def forward(self, X):
        '''
        X shape should be (N, C, H, W)
        '''
        if self.shape is None:
            self.shape = X.shape[find_shape(X, mode= 'features')]
            super().init_params()

        # Vectorized convolution implementation inspired from: https://github.com/slvrfn/vectorized_convolution

        kernel_size, stride, padding =  self.filter_size, self.stride, self.padding
        # get output shape
        B, C_in, H_in, W_in = X.shape
        H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        output_shape = (B, C_in, H_out, W_out)

        # get strided X windows
        strided_X = self.generate_strided_tensor(X, (kernel_size, kernel_size), (stride, stride), (padding, padding), output_shape)

        # convolution with kernels
        # use this to understand: https://ajcr.net/Basic-guide-to-einsum/
        output = np.einsum("nchwkl,ockl->nohw", strided_X, self.parameters['Weight'])

        # add bias if necessary

        output += self.parameters["bias"][np.newaxis, :, np.newaxis, np.newaxis]  

        self.parameters['strided_X'] = strided_X
        self.parameters['X_shape'] = (B, C_in, H_in, W_in)

        
        return output

    def backward(self, dL_dy):
        '''
        dL_dy = gradient of the cost with respect to the output of the conv layer -> (bs, C_out, H, W)

        compute :
        dL_dK = gradient of the cost with respect to the kernels -> (C_out, C_in, kernel_size, kernel_size)
        dL_db = gradient of the cost with respect to the bias -> (C_out)
        dL_dX = gradient of the cost with respect to the input -> (bs, C_in, H_in, W_in)

        '''
        # backpropagation: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
        # backpropagation logic for with strides: https://medium.com/@mayank.utexas/backpropagation-for-convolution-with-strides-8137e4fc2710

        # get parameters
        kernel_size, stride, padding = self.filter_size, self.stride, self.padding

        # compute dL_dK and dL_db
        dL_dW = np.einsum("nchwkl,nohw->ockl", self.parameters['strided_X'], dL_dy) # Convolution(X, dL_dy)
        dL_db = np.einsum('nohw->o', dL_dy) # sum over N, H, W

        # compute dL_dX
        # rotate kernels 180
        kernels_rotated = np.rot90(self.parameters["Weight"], k=2, axes=(2, 3)) # (number of times rotated by 90, k=2)
        # get strided dL_dy windows
        dout_padding = kernel_size - 1 if padding == 0 else kernel_size - 1 - padding
        # dout_padding = 1
        dout_dilate = stride - 1
        # dilate dL_dy based on stride
        if dout_dilate != 0:
            insertion_indices = list(np.arange(1, dL_dy.shape[2]))*dout_dilate
            dL_dy_dilated = np.insert(dL_dy, insertion_indices, values=0, axis=2) # args - input, index, value, axis
            dL_dy_dilated = np.insert(dL_dy_dilated, insertion_indices, values=0, axis=3)

            # Corner Case: in cases where rightmost column and bottommost row gets ignored (due to odd shape and even kernels), these can be added back by extra padding
            new_shape_h = (dL_dy_dilated.shape[2] + 2 * dout_padding - kernel_size) // 1 + 1
            new_shape_w = (dL_dy_dilated.shape[3] + 2 * dout_padding - kernel_size) // 1 + 1
            if (new_shape_h != self.parameters['X_shape'][2]) or (new_shape_w != self.parameters['X_shape'][3]): 
                 # pad incase of size mismatch
                pad_h = self.parameters['X_shape'][2] - new_shape_h 
                pad_w = self.parameters['X_shape'][3] - new_shape_w 
                dL_dy_dilated = np.pad(dL_dy_dilated, ((0,0), (0,0), (0,pad_h), (0,pad_w)))

        else:
            dL_dy_dilated = dL_dy.copy()
        strided_dL_dy = self.generate_strided_tensor(dL_dy_dilated, (kernel_size, kernel_size), (1, 1), (dout_padding, dout_padding), self.parameters['X_shape'], strides=None)
        # compute dL_dX
        dL_dX = np.einsum("nohwkl,ockl->nchw", strided_dL_dy, kernels_rotated) # Convolution(padded dL_dy, kernels_rotated)

        # update parameters
        self.grads['dW'] = dL_dW
        self.grads['db'] = dL_db


        return dL_dX
 


    def generate_strided_tensor(self, inputs, kernel_size, stride, padding, out_shape, strides=None):
        '''
        here kernel_size, stride, padding are tuples of (H, W)
        ''' 
        C_out = inputs.shape[1]
        N, _, H_out, W_out = out_shape

        # pad the input tensor if necessary
        if padding != (0, 0):
            inputs = np.pad(inputs, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode="constant", constant_values=0)
        
            
        # get strides of X
        N_strides, C_out_strides, H_strides, W_strides = inputs.strides if strides is None else strides
        # create a strided tensor
        # use this link to understand: https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
        strided_tensor = np.lib.stride_tricks.as_strided(
            inputs, 
            shape=(N, C_out, H_out, W_out, kernel_size[0], kernel_size[1]), 
            strides=(N_strides, C_out_strides, stride[0] * H_strides, stride[1] * W_strides, H_strides, W_strides))
        return strided_tensor