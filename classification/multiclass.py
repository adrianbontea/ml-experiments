from classification.base import ClassificationExperimentBase
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import pickle


class MulticlassClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        # Scikit-learn is smart enough to detect when you try to use a binary classification algorithm
        # such as SGD on a multiclass classification task (when the labels are not binary) and automatically runs OvA
        # strategy (trains N binary classifiers, one for each class) except for SVM for which it runs OvO
        # (trains N x (N-1)/2 binary classifiers, one between 0 and 1, one between 1 and 2 etc)

        sgd_classifier = SGDClassifier(random_state=77)
        sgd_classifier.fit(training_set_tr, training_labels)

        seven = self.__get_random_digit(training_set_tr, training_labels, 7)
        print("The digit is:", sgd_classifier.predict([seven]))

        # Get the classifier to return the decision scores for each class rather than a prediction
        # The class with the higher score is used for prediction
        scores = sgd_classifier.decision_function([seven])
        print("The decision scores for the digit are:", scores)

        # Can also force Scikit-Learn to use the SGDClassifier with OvO strategy
        ovo = OneVsOneClassifier(sgd_classifier)
        ovo.fit(training_set_tr, training_labels)
        print("OvO: The digit is:", ovo.predict([seven]))

        # Random Forest algorithm can also be used for classification (besides regression - RandomForestRegressor)
        # and is a multiclass algorithm so no need for OvA or OvO strategies
        rnd_forest = RandomForestClassifier()
        rnd_forest.fit(training_set_tr, training_labels)
        print("Random Forest: The digit is:", rnd_forest.predict([seven]))
        print("Random Forest: Probabilities:", rnd_forest.predict_proba([seven]))

        # Evaluate SGD Classifier vs Random Forest based on confusion matrix
        sgd_predictions = cross_val_predict(sgd_classifier, training_set_tr, training_labels, cv=3)
        rnd_forest_predictions = cross_val_predict(rnd_forest, training_set_tr, training_labels, cv=3)

        print("SGD Classifier Confusion Matrix:")
        print(confusion_matrix(training_labels, sgd_predictions))
        print("Random Forest Classifier Confusion Matrix:")
        print(confusion_matrix(training_labels, rnd_forest_predictions))

        # Random Forest generally seems better - higher values on the main diagonal
        # Test persisting a trained classifier
        rnd_forest = RandomForestClassifier()
        rnd_forest.fit(training_set_tr, training_labels)

        file = open("D:\\rnd_forest.dat", "wb")
        pickle.dump(rnd_forest, file)
        file.close()

        file2 = open("D:\\rnd_forest.dat", "rb")
        rnd_forest_2 = pickle.load(file2)
        file2.close()
        print("Random Forest Persisted: The digit is:", rnd_forest_2.predict([seven]))


    def __get_random_digit(self, training_set, labels, digit):
        indexes = np.where(labels == digit)[0]
        return training_set[indexes[np.random.randint(0, len(indexes) - 1)]]