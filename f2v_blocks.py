from tensorflow import keras
from keras.layers import Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

# A ConvBlock in this context performs two things: 1. It performs a convolution on the input image, and 2. It shrinks the spatial dimensions by a factor of 2.
class Face2vecConvBlock(Layer):
    def __init__(self, filter_count : int, *args, **kwargs):
        super(Face2vecConvBlock, self).__init__(*args, **kwargs)

        self.convolutional_layer = Conv2D(filter_count, (3,3), activation="relu", padding="same")
        self.max_pooling = MaxPooling2D((2,2), padding="same")

    def call(self, input):
        x = self.convolutional_layer(input)
        x = self.max_pooling(x)
        return x
    
class Face2vecDeconvBlock(Layer):
    def __init__(self, filter_count : int, *args, **kwargs):
        super(Face2vecDeconvBlock, self).__init__(*args, **kwargs)

        self.deconv_layer = Conv2D(filter_count, (3,3), activation="relu", padding="same")
        self.up_sampling = UpSampling2D((2,2))

    def call(self, input):
        x = self.deconv_layer(input)
        x = self.up_sampling(x)
        return x