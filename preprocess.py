from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

def _process_img(dims : Tuple[int, int]) -> tf.Tensor:
    def __process(image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.central_crop(image, central_fraction=0.5)
        image = tf.image.resize(image, size=dims)
        return image
    return __process

def _make_dataset(dataset_path : str, batch_size : int, dimensions: Tuple[int,int]):
    return tf.data.Dataset\
        .list_files(f"{dataset_path}\\*.jpg", shuffle=True
        ).map(_process_img(dimensions) # Loading
        ).batch(batch_size
        ).map(lambda x: x / 255.0 # Preprocessing
        ).map(lambda x: (x,x) # Needed since fit() doesn't take an explicit y-value when using tf.data.Dataset
        ).prefetch(tf.data.AUTOTUNE)

def make_dataset_split(path : str, batch_size : int, dimensions : Tuple[int,int]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    return (
        _make_dataset(f"{path}\\train", batch_size, dimensions),
        _make_dataset(f"{path}\\test", batch_size, dimensions),
        _make_dataset(f"{path}\\val", batch_size, dimensions))

if __name__=="__main__":
    splitted = make_dataset_split("D:\\Programming\\Machine Learning\\face2vec\\CACD2000", 32, (128,128))

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