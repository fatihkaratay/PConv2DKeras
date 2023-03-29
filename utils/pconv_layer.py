from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D


class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        
        if input_shape[0][channel_axis] is None:
            raise Exception('The channel dimension of the input should be defined. Found "None"')
        
        self.input_dim = input_shape[0][channel_axis]
        
        #Image Kernel
        kernel_shape = self.kernel_shape + (self.input_dim, self.filters)
        self.kernel_mask = self.add_weight(shape=kernel_shape,
                                           initializer=self.kernel_initializer,
                                           name='img_kernel',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        # mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))
        
        # calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)),
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2))
        )
        
        # Window size used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True