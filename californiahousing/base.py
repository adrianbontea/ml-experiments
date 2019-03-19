import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from base import ExperimentBase


class HousingExperimentBase(ExperimentBase):
    __download_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    __housing_path = "datasets/housing"
    __housing_url = __download_url + __housing_path + "/housing.tgz"

    def load_housing_data(self, housing_path=__housing_path):
        csv_path = os.path.join(housing_path, "housing.csv")
        if not os.path.isfile(csv_path):
            self.__fetch_housing_data()
        return pd.read_csv(csv_path)

    def get_train_test(self, data, test_ratio):
        # This ensures the random generated permutation will be the same on each run...
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def __fetch_housing_data(self, housing_url=__housing_url, housing_path=__housing_path):
        if not os.path.isdir(housing_path):
            os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
