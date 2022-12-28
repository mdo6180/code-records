import os
import numpy as np
import types
import requests
import gzip
import shutil

from mlxtend.data import loadlocal_mnist
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    # mnist test dataset: 10,000 samples
    # class 0: 0-979
    # class 1: 980-2114
    # class 2: 2115-3146
    # class 3: 3147-4156
    # class 4: 4157-5138
    # class 5: 5139-6038
    # class 6: 6039-6988
    # class 7: 6989-8016
    # class 8: 8017-8990
    # class 9: 8991-10000

    # mnist train dataset: 60,000 samples

    def __init__(
        self, 
        split: str, 
        classes: list = None, 
        num_samples: int = None, 
        transform: types.ModuleType = None, 
        sorted: bool = False
    ):
    
        """Load mnist data.
        Parameters
        ----------
        split : str
            train or test
        classes : list, optional
            classes to include, by default None
        num_samples : int, optional
            number of samples to include, by default None
        transform : types.ModuleType, optional
            transform module, by default None
        sorted : bool, optional
            return the data sorted, by default False
        Raises
        ------
        ValueError
            if anything other than train and test is specified for split
        """
        # creating folder for mnist dataset
        home_dir = os.path.expanduser("~")
        MNIST_dir = os.path.join(home_dir, "datasets/mnist")

        if os.path.exists(MNIST_dir) is False:
            os.makedirs(MNIST_dir)
        
        # downloading dataset
        def obtain_datasets(url: str, dir: str) -> str:

            compressed_file_path = os.path.join(dir, url.split("/")[-1])
            uncompressed_file_path = compressed_file_path.strip(".gz")

            if os.path.exists(uncompressed_file_path) is False:

                # download dataset from Yann Lecun's website
                r = requests.get(url, allow_redirects=True)
                with open(compressed_file_path, 'wb') as file:
                    file.write(r.content)

                # uncompress dataset
                with gzip.open(compressed_file_path, 'rb') as f_in:
                    with open(uncompressed_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # clean up via removing zipped files
                os.remove(compressed_file_path)
                
            return uncompressed_file_path
            
        train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        train_images_path = obtain_datasets(train_images_url, MNIST_dir)
        
        train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        train_labels_path = obtain_datasets(train_labels_url, MNIST_dir)

        test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        test_images_path = obtain_datasets(test_images_url, MNIST_dir)

        test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        test_labels_path = obtain_datasets(test_labels_url, MNIST_dir)

        if split == "test":
            images_path = test_images_path
            labels_path = test_labels_path
        elif split == "train":
            images_path = train_images_path
            labels_path = train_labels_path
        else:
            raise ValueError("Value entered into 'split' argument must be either 'train' or 'test'")

        X, Y = loadlocal_mnist(
            images_path=images_path,
            labels_path=labels_path,
        )

        # ---------------- add additional filters here -----------------
        # filter out undesired classes
        if classes != None:
            mask = []
            for y in Y:
                if y in classes:
                    mask.append(True)
                else:
                    mask.append(False)
            X = X[mask]
            Y = Y[mask]
        
        # sort classes from smallest to largest
        if sorted==True:
            X = X[Y.argsort()]
            Y.sort()

        # select first n samples
        if num_samples is not None:
            X = X[:num_samples, :]
            Y = Y[:num_samples]
        # ---------------- add additional filters here -----------------
        
        self.x = X
        self.y = Y
        self.num_samples = self.y.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        IMG_SHAPE = (28,28)
        image = np.reshape(self.x[index], IMG_SHAPE)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        label = self.y[index]
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return self.num_samples

def view_image(image, mode: str) -> None:
    import matplotlib.pyplot as plt

    img = image.numpy()
    img = img.squeeze()

    if mode is "show":
        plt.imshow(img, interpolation='nearest')
        plt.show()
    elif mode is "save":
        pass


if __name__ == "__main__":
    mnist_test = MNISTDataset(split="test")
    img, label = mnist_test[0]
