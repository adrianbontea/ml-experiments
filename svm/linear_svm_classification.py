from base import ExperimentBase
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions


class LinearSvmClassificationExperiment(ExperimentBase):
    async def do_run_async(self):
        iris = load_iris()

        # Training with just 2 features
        X = iris.data[:, 2:4]  # Petal length, Petal Width - Shape (150,2)

        # Binary classification
        y = (iris.target == 2).astype(np.int)  # Iris-Virginica

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X)

        # Plot Scaled Data Set with binary labels - Virginica and Non-Virginica instances
        colors = ['green', 'blue']
        plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y, cmap=ListedColormap(colors))
        plt.show()

        svm_classifier = LinearSVC(C=1, loss="hinge")
        svm_classifier.fit(X_tr, y)

        # Plot decision regions for binary labels
        plot_decision_regions(X_tr, y, svm_classifier)
        plt.xlabel('Petal Length [cm]')
        plt.ylabel('Petal Width [cm]')
        plt.title('SVM on Iris - 2 Features - Binary Classes')
        plt.show()

        print(f"A flower with petal length = 5.5 and petal width = 1.7 is Iris-Virginica? {svm_classifier.predict([[5.5, 1.7]])}")

        # Multiclass classification?
        y = iris.target
        X_tr = scaler.fit_transform(X)

        # Plot Scaled Data Set with multiclass
        colors = ['green', 'blue', 'red', 'yellow']
        plt.scatter(X_tr[:, 0], X_tr[:, 1], c=y, cmap=ListedColormap(colors))
        plt.show()

        svm_classifier = LinearSVC(C=1, loss="hinge")
        svm_classifier.fit(X_tr, y)

        # Plot decision regions for multiclass
        plot_decision_regions(X_tr, y, svm_classifier)
        plt.xlabel('Petal Length [cm]')
        plt.ylabel('Petal Width [cm]')
        plt.title('SVM on Iris - 2 Features - Multi Classes')
        plt.show()

        print(f"A flower with petal length = 5.5 and petal width = 1.7 is:{iris.target_names[svm_classifier.predict([[5.5, 1.7]])[0]]}")

        # Training with all 4 features
        X = iris.data  # Sepal Length, Sepal Width, Petal length, Petal Width - Shape (150,4)

        # Binary classification
        y = (iris.target == 2).astype(np.int)  # Iris-Virginica
        X_tr = scaler.fit_transform(X)

        svm_classifier = LinearSVC(C=1, loss="hinge")
        svm_classifier.fit(X_tr, y)
        print(f"A flower with sepal length = 1.3, sepal width = 2.2, petal length = 5.5 and petal width = 1.7 is Iris-Virginica?{svm_classifier.predict([[1.3, 2.2, 5.5, 1.7]])}")

        # Plot decision regions for binary labels when multiple features?
