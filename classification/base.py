from mnist import *
from base.base import ExperimentBase


class ClassificationExperimentBase(ExperimentBase):
    __mnist_path = "datasets/mnist"

    def load_train_images(self):
        return self.__load_file("train-images-idx3-ubyte.gz")

    def load_train_labels(self):
        return self.__load_file("train-labels-idx1-ubyte.gz")

    def load_test_images(self):
        return self.__load_file("t10k-images-idx3-ubyte.gz")

    def load_test_labels(self):
        return self.__load_file("t10k-labels-idx1-ubyte.gz")

    def __load_file(self, filename):
        gz_path = os.path.join(self.__mnist_path, filename)
        if not os.path.isfile(gz_path):
            self.__download_mnist_file(filename)
        fd = gzip.open(gz_path)
        return parse_idx(fd)

    def __download_mnist_file(self, filename):
        if not os.path.isdir(self.__mnist_path):
            os.makedirs(self.__mnist_path)
        download_file(filename, self.__mnist_path)
