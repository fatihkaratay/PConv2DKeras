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
        
    def call(self, inputs, mask=None):
        # both image and mask shouold be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConv2D must be called on a list of two tensors [img, mask]. Instead got: ' - str(inputs))
        
        # padding done explicitly so that padding becomes part of the masked partial convolutional
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)
        
        # apply convolutional to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask,
            strides=self.strides,
            padding = 'valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        
        # apply convolutional to image
        img_output = K.conv2d(
            (images*masks), 
            self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        
        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)
        
        # clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)
        
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output
        
        # Normalize image output
        img_output = img_output * mask_ratio
        
        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format
            )
            
        # Apply activations on the image
        if self. activation is not None:
            img_output = self.activation(img_output)
        
        return [img_output, mask_output]