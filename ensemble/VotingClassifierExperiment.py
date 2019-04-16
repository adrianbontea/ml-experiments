from classification import ClassificationExperimentBase
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class VotingClassifierExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()
        training_labels = super().load_train_labels()

        test_set = super().load_test_images()
        test_labels = super().load_test_labels()

        # Training and test sets need to be reshaped from 3D (m,28,28) to 2D (m, 784) for the classifiers to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        test_set_tr = test_set.reshape((10000, 784))

        # Hard Voting (The class with the higher number of votes is output)
        sgd_clf = SGDClassifier()
        rnd_clf = RandomForestClassifier()
        k_clf = KNeighborsClassifier()  # Note: training this is very slow on the MNIST data set

        voting_clf = VotingClassifier(
            estimators=[('sgd', sgd_clf), ('rf', rnd_clf), ('k', k_clf)],
            voting='hard'
        )

        # Compute and compare the accuracy score. The voting classifier should get an accuracy score better than each individual
        for clf in [sgd_clf, rnd_clf, k_clf, voting_clf]:
            clf.fit(training_set_tr, training_labels)
            predictions = clf.predict(test_set_tr)
            print(type(clf).__name__, accuracy_score(test_labels, predictions))

        # Soft Voting (The class with the highest probability averaged across all classifiers is output)
        # All classifiers in the ensemble need to be able to predict probabilities (predict_proba)
        voting_clf.voting = 'soft'
