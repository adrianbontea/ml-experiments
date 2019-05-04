from .base import ClassificationExperimentBase
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

class MultioutputClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        # Group training instances and labels into pairs of two
        training_set_tr_pairs = training_set_tr.reshape((30000, 1568))
        training_labels_pairs = training_labels.reshape((30000, 2))

        k_neigh_classifier = KNeighborsClassifier()
        k_neigh_classifier.fit(training_set_tr_pairs, training_labels_pairs)

        # Get 2 random digits and merge them into the same vectorized image
        six = super().get_random_digit(training_set_tr, training_labels, 6)
        three = super().get_random_digit(training_set_tr, training_labels, 3)

        six_and_three = np.append(six, three)
        print(f"K-Neighbors prediction for vectorized combined image of 6 and 3 is:{k_neigh_classifier.predict([six_and_three])}")

        # RandomForestClassifier also works with multi(label/output) classification (multi-dimensional training labels)
        # SGD doesn't
        rnd_forest_classifier = RandomForestClassifier()
        rnd_forest_classifier.fit(training_set_tr_pairs, training_labels_pairs)

        print(f"Random Forest prediction for vectorized combined image of 6 and 3 is:{rnd_forest_classifier.predict([six_and_three])}")

        # Attempt to evaluate and compare K_Neighbours and Random Forest classifiers based on confusion matrix
        # in multioutput classification task using the test set with each 2 images merged
        # Note predict on the whole transformed set takes forever so trying single predictions in a loop...
        test_set_tr = super().load_test_images().reshape((5000, 1568))
        test_labels_tr = super().load_test_labels().reshape((5000, 2))

        k_neigh_predictions = []
        rnd_forest_predictions = []

        for instance in test_set_tr:
            k_neigh_predictions.append(k_neigh_classifier.predict([instance])[0])
            rnd_forest_predictions.append(rnd_forest_classifier.predict([instance])[0])

        k_neigh_predictions_nd = np.array(k_neigh_predictions)
        rnd_forest_predictions_nd = np.array(rnd_forest_predictions)

        # As expected, confusion_matrix function will fail with message "multilabel-multioutput" is not supported!
        print("Confusion matrix for K-Neighbours")
        print(confusion_matrix(test_labels_tr, k_neigh_predictions_nd))

        print("Confusion matrix for Random Forest")
        print(confusion_matrix(test_labels_tr, rnd_forest_predictions_nd))

