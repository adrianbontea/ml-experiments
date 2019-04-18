import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict

from .base import ClassificationExperimentBase


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
        some_digit = training_set_tr[index_of_five, ]
        super().show_digit(some_digit)

        print("some_digit is a 5:", sgd_classifier.predict([some_digit]))

        # Pick a not five
        index_of_not_five = self.__get_index_of_random_false(five_binary_labels)
        some_digit = training_set_tr[index_of_not_five, ]
        super().show_digit(some_digit)

        print("some_digit is a 5:", sgd_classifier.predict([some_digit]))

        # Evaluate the current SGD classifier
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
        # Precision-Recall Tradeoff...
        # Increasing the threshold increases precision and decreases recall
        # Conversely, decreasing threshold increases recall and decreases precision

        sgd_classifier = SGDClassifier(random_state=77)
        sgd_classifier.fit(training_set_tr, five_binary_labels)

        # Pick a five
        index_of_five = self.__get_index_of_random_true(five_binary_labels)
        some_digit = training_set_tr[index_of_five, ]

        score = sgd_classifier.decision_function([some_digit])
        threshold = 0

        print(score > threshold)  # Outputs True when threshold is 0

        threshold = 300000

        print(score > threshold)  # Outputs False when increasing threshold

        sgd_classifier = SGDClassifier(random_state=77)
        # Get the scores of all instances in the training set using K-fold predictions
        scores = cross_val_predict(sgd_classifier, training_set_tr, five_binary_labels, cv=3, method="decision_function")

        # Now compute the Precision-Recall curves for each possible thresholds
        precisions, recalls, thresholds = precision_recall_curve(five_binary_labels, scores)
        # Get list of thresholds that would result in a precision of at least 88% and recall of at least 70%
        # Note ndarrays of precision and recall (dimension 1) are one element larger than thresholds
        # so we select all but last element

        thresholds_for_precision_and_recall = self.__get_thresholds_for(precisions[:-1], recalls[:-1], thresholds, 0.88, 0.70)
        print("Thresholds for Precision at least 0.88 and Recall at least 0.7 are:", thresholds_for_precision_and_recall)

        # We get 33 possible threshold values for which the precision would be at least 88% and recall at least 70%
        # Test one of them
        test_threshold = thresholds_for_precision_and_recall[np.random.randint(0, len(thresholds_for_precision_and_recall) - 1)]
        predictions = scores > test_threshold

        print("Precision:", precision_score(five_binary_labels, predictions))
        print("Sensitivity", recall_score(five_binary_labels, predictions))


    def __get_index_of_random_true(self, binary_labels):
        indexes_of_trues = np.where(binary_labels == True)[0]
        return indexes_of_trues[np.random.randint(0, len(indexes_of_trues) - 1)]

    def __get_index_of_random_false(self, binary_labels):
        indexes_of_falses = np.where(binary_labels == False)[0]
        return indexes_of_falses[np.random.randint(0, len(indexes_of_falses) - 1)]

    def __get_thresholds_for(self, precisions, recalls, thresholds, precision_at_least, recall_at_least):
        indexes_precisions = np.where(precisions > precision_at_least)[0]
        indexes_recalls = np.where(recalls > recall_at_least)[0]
        intersection_indexes = np.intersect1d(indexes_precisions, indexes_recalls)

        return thresholds[intersection_indexes]


