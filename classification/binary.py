from classification.base import ClassificationExperimentBase
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class BinaryClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        # Transform the labels to an array of binary labels (5 or not 5)
        five_binary_labels = (training_labels == 5)
        sgd_classifier = SGDClassifier(random_state=77)
        sgd_classifier.fit(training_set_tr, five_binary_labels)

        # Pick a five
        index_of_five = self.__get_index_of_random_true(five_binary_labels)
        some_digit = training_set_tr[index_of_five]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")

        print("some_digit is a 5:", sgd_classifier.predict([some_digit]))

        # Pick a not five
        index_of_not_five = self.__get_index_of_random_false(five_binary_labels)
        some_digit = training_set_tr[index_of_not_five]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")

        print("some_digit is a 5:", sgd_classifier.predict([some_digit]))

    def __get_index_of_random_true(self, binary_labels):
        indexes_of_trues = np.where(binary_labels == True)[0]
        return indexes_of_trues[np.random.randint(0, len(indexes_of_trues) - 1)]

    def __get_index_of_random_false(self, binary_labels):
        indexes_of_falses = np.where(binary_labels == False)[0]
        return indexes_of_falses[np.random.randint(0, len(indexes_of_falses) - 1)]