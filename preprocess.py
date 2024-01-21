from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

def process_img(dims : Tuple[int, int]) -> tf.Tensor:
    def __process(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.central_crop(image, central_fraction=0.5)
        image = tf.image.resize(image, size=dims)
        return image
    return __process

def make_dataset(dataset_path : str, batch_size : int, dimensions: Tuple[int,int]):
    return tf.data.Dataset\
        .list_files(f"{dataset_path}\\*.jpg", shuffle=True
        ).map(process_img(dimensions) # Loading
        ).batch(batch_size
        ).map(lambda x: x / 255.0 # Preprocessing
        ).prefetch(tf.data.AUTOTUNE)

def train_test_split(train_percent : float, ds: tf.data.Dataset, ds_size : int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    test_plus_val_percent = 1.0 - train_percent
    train_size = int(train_percent * ds_size)
    test_size = int(test_plus_val_percent * ds_size / 2.0)
    val_size = int(test_plus_val_percent * ds_size / 2.0)

    ds = ds.shuffle(1024)
    train = ds.take(train_size)
    test = ds.skip(train_size)
    val = test.skip(val_size)
    test = test.take(test_size)
    return (train, test, val)

if __name__=="__main__":
    datapoints = len([filename for filename in glob.glob("D:\\Programming\\Machine Learning\\face2vec\\CACD2000\\*.jpg")])
    ds = make_dataset("D:\\Programming\\Machine Learning\\face2vec\\CACD2000", 32, (128,128))

    splitted = train_test_split(0.7, ds, datapoints)

    img = next(iter(splitted[0].take(1)))[0]
    plt.imshow(img)
    plt.show()

    img = next(iter(splitted[1].take(1)))[0]
    plt.imshow(img)
    plt.show()

    img = next(iter(splitted[2].take(1)))[0]
    plt.imshow(img)
    plt.show()

    print("H")
    """for batch in ds.take(1):
        for img in batch:
            plt.imshow(img)
            plt.show()"""