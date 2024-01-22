from face2vec import *
from tensorflow import keras
from keras.layers import Input
from keras import Model
from preprocess import make_dataset_split

BATCH_SIZE = 128

if __name__=="__main__":
    print("GPUS: ", tf.config.list_physical_devices('GPU'))
    encoder = Face2vecEncoder((128,128), 512, 3)
    decoder = Face2VecDecoder(encoder)

    # Build the model
    input_layer = Input(shape=(128,128,3))
    signature = encoder(input_layer)
    decoded = decoder(signature)
    model = Model(input_layer, decoded)
    model.summary()

    print("Loading data...")
    dataset = make_dataset_split("D:\\Programming\\Machine Learning\\face2vec\\CACD2000", BATCH_SIZE, (128,128))
    x_train = dataset[0]
    x_test = dataset[1]

    print("Training model")
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train,
              epochs=50,
              batch_size=BATCH_SIZE,
              validation_data=x_test,
              callbacks=[callback])
    model.save("saved_model")