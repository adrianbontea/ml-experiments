from sklearn.tree import DecisionTreeClassifier
from classification import ClassificationExperimentBase
from sklearn.ensemble import AdaBoostClassifier
import time


class AdaBoostingExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        # How AdaBoost works:
        # Each instance weight w(i) is initially set to 1/m so given equal weight (m is the number of instances)
        # A first predictor is trained and its WEIGHTED ERROR RATE r1 is computed on the training set:
        # rj = Sum [i=1-> m where Yj^(i) != y(i)] (w(i)) / Sum [i=1 -> m] (w(i)) (yj^(i) is the jth predictor prediction for instance i)

        # Next the predictor's weight is then computed:
        # aj = n * log((1 - rj)/rj)  (n is the learning rate)
        # In machine learning, the logarithm or exponential without a base specified often refers to the natural log/exp so base is e

        # Next the instance weights are updated and the misclassified instances are boosted:
        # for i = 1 -> m, if y^j(i) == y(i) w(i) -> w(i), otherwise w(i) -> w(i) * exp (aj)

        # Finally, a new predictor is trained using the updated weights, and the whole process is
        # repeated (the new predictor’s weight is computed, the instance weights are updated,
        # then another predictor is trained, and so on). The algorithm stops when the desired
        # number of predictors is reached, or when a perfect predictor is found.

        # To make predictions, AdaBoost simply computes the predictions of all the predictors
        # and weighs them using the predictor weights αj. The predicted class is the one that
        # receives the majority of weighted votes:
        # y^(x) = argmax k Sum [i=1->N where y^j(x) = k] aj (N is the number of predictors)

        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        # 60 000 arrays of 784 Pixel intensities from 0 to 255! (Grayscale 8-bit image)
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1), n_estimators=10,
            algorithm="SAMME.R", learning_rate=0.5)

        start_time = time.time()
        ada_clf.fit(training_set_tr, training_labels)
        elapsed = time.time() - start_time

        print(f"Training an AdaBoostClassifier with 10 stumps took {elapsed} seconds")

        # Scikit-Learn actually uses a multiclass version of AdaBoost called SAMME which
        # stands for Stagewise Additive Modeling using a Multiclass Exponential loss function.
        # In this case since the DecisionTree estimator/predictor can estimate class probabilities (exposes a predict_proba method)
        # Scikit-Learn can use a variant of SAMME called SAMME.R (the R stands
        # for “Real”), which relies on class probabilities rather than predictions and generally
        # performs better.

        # A Decision Tree with max_depth=1, in other words, a tree composed of a single decision node plus two leaf nodes.
        # is called a Decision Stump

        an_eight = super().get_random_digit(training_set_tr, training_labels, 8)
        super().show_digit(an_eight)

        print(f"AdaBoostClassifier prediction is: {ada_clf.predict([an_eight])}")
