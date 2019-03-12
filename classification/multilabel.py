import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from classification.base import ClassificationExperimentBase


class MultilabelClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        # Introduce just 2 classes (as opposed to 9 for 9 digits): large digits and even digits
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

        print("K-Neighbors prediction for 8 is:", classifier.predict([eight]))
        print("K-Neighbors prediction for 9 is:", classifier.predict([nine]))

        # Try with 9 classes and 9 binary labels (one for each digit)
        zeros = (training_labels == 0)
        ones = (training_labels == 1)
        twos = (training_labels == 2)
        threes = (training_labels == 3)
        fours = (training_labels == 4)
        fives = (training_labels == 5)
        sixes = (training_labels == 6)
        sevens = (training_labels == 7)
        eights = (training_labels == 8)
        nines = (training_labels == 9)

        multi_labels = np.append(zeros.reshape((60000, 1)), ones.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, twos.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, threes.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, fours.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, fives.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, sixes.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, sevens.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, eights.reshape((60000, 1)), axis=1)
        multi_labels = np.append(multi_labels, nines.reshape((60000, 1)), axis=1)

        k_neigh_classifier = KNeighborsClassifier()
        k_neigh_classifier.fit(training_set_tr, multi_labels)

        six = super().get_random_digit(training_set_tr, training_labels, 6)
        five = super().get_random_digit(training_set_tr, training_labels, 5)

        print("K-Neighbors prediction for 6 is:", k_neigh_classifier.predict([six]))
        print("K-Neighbors prediction for 5 is:", k_neigh_classifier.predict([five]))

        rnd_forest_classifier = RandomForestClassifier()
        rnd_forest_classifier.fit(training_set_tr, multi_labels)

        print("Random Forest prediction for 6 is:", rnd_forest_classifier.predict([six]))
        print("Random Forest prediction for 5 is:", rnd_forest_classifier.predict([five]))
