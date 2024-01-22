from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Layer, Flatten, Dense, Reshape, Conv2D, Input, UpSampling2D
from f2v_blocks import Face2vecConvBlock, Face2vecDeconvBlock

class Face2vecEncoder(Layer):
    def __init__(self, image_dims : Tuple[int,int], output_dims : int, conv_blocks: int, *args, **kwargs):
        super(Face2vecEncoder, self).__init__(*args, **kwargs)
        flatten_dims = ((image_dims[0] * image_dims[1]) // (2 ** (2*conv_blocks)))
        if output_dims > flatten_dims:
            tf.compat.v1.logging.log(tf.compat.v1.logging.WARN, f"Flatten will produce a '{flatten_dims}' unit vector after '{conv_blocks}', so a '{output_dims}' unit output risks being too large")
        
        self.input_dimensions = image_dims
        self.convolution_blocks = [Face2vecConvBlock(32 * (2**i)) for i in range(conv_blocks)]
        self.flatten = Flatten()
        self.dense = Dense(output_dims, activation="relu")
    
    def call(self, input):
        x = input
        for block in self.convolution_blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def compute_flatten_shape(self, input_shape):
        shape = input_shape
        for block in self.convolution_blocks:
            shape = block.compute_output_shape(shape)
        return shape, self.flatten.compute_output_shape(shape)
    
    def model(self) -> Model:
        x = Input(shape=(self.input_dimensions[0], self.input_dimensions[1], 3))
        return Model(inputs=[x], outputs=self.call(x))


class Face2VecDecoder(Layer):
    def __init__(self, encoder : Face2vecEncoder, *args, **kwargs):
        super(Face2VecDecoder, self).__init__(*args, **kwargs)

        encoder_layers = encoder.model().layers
        f2v_encoder_outputs = encoder.compute_flatten_shape(encoder_layers[0].input_shape[0])
        deconv_count = len(encoder.convolution_blocks)

        self.encoder_dense_output_shape = encoder_layers[-1].output_shape[-1]
        
        self.inv_dense = Dense(f2v_encoder_outputs[1][1:][0], activation="relu")
        self.reshape = Reshape(tuple(f2v_encoder_outputs[0][1:]))
        self.deconv_blocks = [Face2vecDeconvBlock(32 * (2**i)) for i in reversed(range(deconv_count - 1))]
        self.recreated_image = Conv2D(3, (3,3), activation="sigmoid", padding="same")
        self.upsampled = UpSampling2D((2,2))
    
    def call(self, inputs):
        x = self.inv_dense(inputs)
        x = self.reshape(x)
        for block in self.deconv_blocks:
            x = block(x)
        x = self.recreated_image(x)
        x = self.upsampled(x)
        return x
    
    def model(self) -> Model:
        x = Input(shape=(self.encoder_dense_output_shape))
        return Model(inputs=[x], outputs=self.call(x))