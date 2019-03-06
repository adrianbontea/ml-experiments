from classification.base import ClassificationExperimentBase
from sklearn.linear_model import SGDClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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

        # Evaluate the current SGD regressor
        y_train_pred = cross_val_predict(sgd_classifier, training_set_tr, five_binary_labels, cv=3)
        cm = confusion_matrix(five_binary_labels, y_train_pred)
        print("Confusion Matrix for random_state = 77:", cm)

        # For random_state = 77 we get
        # [[54168   411]
        # [ 1762  3659]]
        # Each row represents an actual class (being a binary classifier only two classes: 0 and 1 top-down)
        # and each column represents a predicted class (0 and 1 left-right)
        # That is 54168 true negatives, 411 false positives, 1762 false negatives and 3659 true positives

        # Precision (TP/(TP+FP)
        print("Precision for random_state = 77:", precision_score(five_binary_labels, y_train_pred))

        # Recall (Sensitivity) (TP/(TP+FN)
        print("Sensitivity for random_state = 77:", recall_score(five_binary_labels, y_train_pred))

        # f1 mean (harmonic mean between Precision and Sensitivity)
        print("f1 mean for random_state = 77:", f1_score(five_binary_labels, y_train_pred))

        # Precision and Sensitivity are 1 for an IDEAL classifier
        # That is when both FP and FN are 0 (in other words no mistakes)

        # Let's try with a lower random_state
        sgd_classifier = SGDClassifier(random_state=42)

        y_train_pred = cross_val_predict(sgd_classifier, training_set_tr, five_binary_labels, cv=3)
        cm = confusion_matrix(five_binary_labels, y_train_pred)

        print("Confusion Matrix for random_state = 42:", cm)
        print("Precision for random_state = 42:", precision_score(five_binary_labels, y_train_pred))
        print("Sensitivity for random_state = 42:", recall_score(five_binary_labels, y_train_pred))
        print("f1 mean for random_state = 42:", f1_score(five_binary_labels, y_train_pred))

        # We get lower precision (more false positives) but higher sensitivity (less false negatives)

    def __get_index_of_random_true(self, binary_labels):
        indexes_of_trues = np.where(binary_labels == True)[0]
        return indexes_of_trues[np.random.randint(0, len(indexes_of_trues) - 1)]

    def __get_index_of_random_false(self, binary_labels):
        indexes_of_falses = np.where(binary_labels == False)[0]
        return indexes_of_falses[np.random.randint(0, len(indexes_of_falses) - 1)]