from classification.base import ClassificationExperimentBase
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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

        classifier = KNeighborsClassifier()
        classifier.fit(training_set_tr_pairs, training_labels_pairs)

        # Get 2 random digits and merge them into the same vectorized image
        six = super().get_random_digit(training_set_tr, training_labels, 6)
        three = super().get_random_digit(training_set_tr, training_labels, 3)

        six_and_three = np.append(six, three)
        print("K-Neighbors prediction for vectorized combined image is:", classifier.predict([six_and_three]))

        # RandomForestClassifier also works with multi(label/output) classification (multi-dimensional training labels)
        # SGD doesn't
        classifier = RandomForestClassifier()
        classifier.fit(training_set_tr_pairs, training_labels_pairs)

        print("Random Forest prediction for vectorized combined image is:", classifier.predict([six_and_three]))
