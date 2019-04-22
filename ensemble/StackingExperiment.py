from classification import ClassificationExperimentBase
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier


class StackingExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        # Stacking is based on a simple idea: instead of using trivial functions
        # (such as hard voting for classification or average for regression)
        # to aggregate the predictions of all predictors in an ensemble,
        # why don’t we train a model to perform this aggregation?

        # The simplest stacking model involves one layer of predictors and a single aggregating predictor on top called a blender.
        # First, the training set is split in two subsets. The first subset is used to train the predictors in the first layer.
        # Next, the second subset is used with the predictors to make clean predictions (predictors never saw the test instances)
        # This results in n * m predictions where n is the number of predictors in the first layer and m is the size of the second subset.
        # Finally, these predictions and m labels for the second subset are used to train the blender (m instances of n features each)
        # After the training phase, to make a prediction for a new instance, the ensemble will be fed the new instance
        # Starting from the bottom layer which will result in a new n-features instance which will be fed to the blender
        # to make a final prediction.
        # It is also possible to create multiple layers of blenders up to a single final blender.

        # Scikit learn doesn't support Stacking out of the box but it's easy to create a custom implementation
        # or use existing extensions such as the one in mlxtend modules

        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        # 60 000 arrays of 784 Pixel intensities from 0 to 255! (Grayscale 8-bit image)
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        classifiers = []

        for i in range(0, 3):
            classifiers.append(DecisionTreeClassifier())

        blender = DecisionTreeClassifier()

        # Obviously, the predictors and the blender can be different type of classifiers

        stacking_clf = StackingClassifier(classifiers=classifiers, meta_classifier=blender)
        stacking_clf.fit(training_set_tr, training_labels)

        a_six = super().get_random_digit(training_set_tr, training_labels, 6)
        print(f"StackingClassifier prediction is: {stacking_clf.predict([a_six])}")
