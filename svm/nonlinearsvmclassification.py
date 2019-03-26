from base import ExperimentBase
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class NonLinearSvmClassificationExperiment(ExperimentBase):
    async def do_run_async(self):
        # A random dataset returning 100 sample instances with 2 features and 100 binary labels (1 or 0)
        # The instances are not lineary separable like the iris dataset in order to test polynomial SVM classification
        moons = make_moons()
        X = moons[0]
        y = moons[1]

        # Plot Original Data Set
        colors = ['green', 'blue']
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors))
        plt.show()

        pipeline = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),  # Polynomial degree is usually number of features + 1?
            ("scaler", StandardScaler())
        ])
        X_tr = pipeline.fit_transform(X)

        classifier = LinearSVC(C=10, loss="hinge", max_iter=10000)
        classifier.fit(X_tr, y)

        random_index = np.random.randint(0, 100)
        print("Non-linear SVM prediction:", classifier.predict([X_tr[random_index, ]]))
        print("Label:", y[random_index])

        # Adding polynomial features is simple to implement and can work great with all sorts
        # of Machine Learning algorithms (not just SVMs), but at a low polynomial degree it
        # cannot deal with very complex datasets, and with a high polynomial degree it creates
        # a huge number of features, making the model too slow.

        # Using Polynomial Kernel
        # It makes it possible to
        # get the same result as if you added many polynomial features, even with very high degree
        # polynomials, without actually having to add them. So there is no combinatorial
        # explosion of the number of features since you don’t actually add any features. This
        # trick is implemented by the SVC class

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X)

        classifier = SVC(kernel="poly", degree=3, coef0=1, C=5)
        classifier.fit(X_tr, y)

        # Plot decision regions
        plot_decision_regions(X_tr, y, classifier)
        plt.title('3 Degree Polynomial Kernel SVM Classifier on Moons')
        plt.show()

        classifier = SVC(kernel="poly", degree=10, coef0=1, C=5)
        classifier.fit(X_tr, y)

        # Plot decision regions
        plot_decision_regions(X_tr, y, classifier)
        plt.title('10 Degree Polynomial Kernel SVM Classifier on Moons')
        plt.show()

        # Let's try other classifiers without the polynomial features (just the original 2) so we can plot
        # their decision regions and compare

        # Try Random Forest Classifier and decision regions
        classifier = RandomForestClassifier()
        classifier.fit(X_tr, y)

        # Plot decision regions
        plot_decision_regions(X_tr, y, classifier)
        plt.title('Random Forest Classifier on Moons')
        plt.show()

        print("Random Forest prediction:", classifier.predict([X_tr[random_index, ]]))
        print("Label:", y[random_index])

        # Try K-Nearest Neighbours Classifier and decision regions
        classifier = KNeighborsClassifier()
        classifier.fit(X_tr, y)

        # Plot decision regions
        plot_decision_regions(X_tr, y, classifier)
        plt.title('K-Nearest Neighbours Classifier on Moons')
        plt.show()

        print("K-Nearest Neighbours prediction:", classifier.predict([X_tr[random_index, ]]))
        print("Label:", y[random_index])
