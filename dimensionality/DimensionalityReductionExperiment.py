from classification import ClassificationExperimentBase
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import time
from sklearn.neighbors import KNeighborsClassifier


class DimensionalityReductionExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        # Principal Component Analysis (PCA) is by far the most popular dimensionality reduction
        # algorithm. First it identifies the hyperplane that lies closest to the data, and then
        # it projects the data onto it.

        # How PCA works: For the number of dimensions that you want to reduce a dataset to, it identifies the axis
        # for which the projection of the dataset onto generates the maximum amount of variance (or the axis that minimizes the mean squared distance
        # between the original dataset and its projection onto that axis based on Pythagoras' theorem)
        # It starts with a first axis then finds a second axis orthogonal to the first that maximizes the amount of remaining variance
        # and then a third axis orthogonal to the first two and so on - as many axes as the number of dimensions required to reduce the dataset to.
        # The vectors that define the axis are called Principal Components
        # Once you have identified all the principal components, you can reduce the dimensionality
        # of the dataset down to d dimensions by projecting it onto the hyperplane
        # defined by the first d principal components.

        training_set = super().load_train_images()

        # Training set needs to be reshaped from 3D (60000,28,28) to 2D (60000, 784) for the classifier to be able to
        # use in training phase
        training_set_tr = training_set.reshape((60000, 784))
        training_labels = super().load_train_labels()

        X = training_set_tr[:1000, :]  # First 1000 instances

        # There is a standard matrix factorization technique called Singular Value Decomposition (SVD)
        # that can decompose the training set matrix X into the dot product of three matrices U
        # · Σ · VT, where VT contains all the principal components that we are looking for.

        X_centered = X - X.mean(axis=0)
        U, s, V = np.linalg.svd(X_centered)

        # The principal components vectors are then the columns of the transpose of V matrix
        C1 = V.T[:, 0]   # Shape (784,)
        C2 = V.T[:, 1]   # Shape (784,)

        # To project the training set onto the hyperplane, you can simply compute the dot
        # product of the training set matrix X by the matrix Wd, defined as the matrix containing the first d principal components
        # (i.e., the matrix composed of the first d columns of VT)

        W2 = V.T[:, :2]
        X2D = X_centered.dot(W2)

        # Same using Scikit-Learn
        pca = PCA(n_components=2)
        X2D = pca.fit_transform(X)  # X2D should be identical to the one computed above?

        # Instead of arbitrarily choosing the number of dimensions to reduce down to, it is
        # generally preferable to choose the number of dimensions that add up to a sufficiently
        # large portion of the variance (e.g., 95%). Unless, of course, you are reducing dimensionality
        # for data visualization—in that case you will generally want to reduce the
        # dimensionality down to 2 or 3.

        pca = PCA(n_components=0.95)
        X_reduced = pca.fit_transform(X)  # The 1000 instances should now have 129 features instead of the original 784

        # One problem with the preceding implementation of PCA is that it requires the whole
        # training set to fit in memory in order for the SVD algorithm to run. Fortunately,
        # Incremental PCA (IPCA) algorithms have been developed: you can split the training
        # set into mini-batches and feed an IPCA algorithm one mini-batch at a time. This is
        # useful for large training sets, and also to apply PCA online (i.e., on the fly, as new
        # instances arrive).

        X = training_set_tr

        n_batches = 100  # 100 batches of 600 instances
        inc_pca = IncrementalPCA(n_components=129)
        for X_batch in np.array_split(X, n_batches):
            inc_pca.partial_fit(X_batch)

        X_mnist_reduced = inc_pca.transform(X)

        # Measure the difference in the time required to train a K-Neighbors Classifier (known to be slow)
        # on the original and reduced MNIST dataset...The difference should be huge!

        start_time = time.time()
        clf = KNeighborsClassifier()
        clf.fit(X, training_labels)
        elapsed = time.time() - start_time

        print(f"Training a K-Neighbors Classifier on the original MNIST dataset took {elapsed} seconds.")

        start_time = time.time()
        clf.fit(X_mnist_reduced, training_labels)
        elapsed = time.time() - start_time

        print(f"Training a K-Neighbors Classifier on the reduced MNIST dataset took {elapsed} seconds.")






