from classification.base import ClassificationExperimentBase
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


class MultilabelClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        # Labels for large digits
        large_digits = (training_labels > 7)
        # Labels for even digits
        even_digits = (training_labels % 2 == 0)

        # Double labels (pairs)
        double_labels = np.append(large_digits.reshape((60000, 1)), even_digits.reshape((60000, 1)), axis=1)

        classifier = KNeighborsClassifier()
        classifier.fit(training_set_tr, double_labels)

        eight = super().get_random_digit(training_set_tr, training_labels, 8)
        nine = super().get_random_digit(training_set_tr, training_labels, 9)

        print("Prediction for 8 is:", classifier.predict([eight]))
        print("Prediction for 9 is:", classifier.predict([nine]))
