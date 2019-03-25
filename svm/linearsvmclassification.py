from base import ExperimentBase
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class LinearSvmClassificationExperiment(ExperimentBase):
    async def do_run_async(self):
        iris = load_iris()
        X = iris.data[:, 2:4]  # Petal length, Petal Width - Shape (150,2)

        # Binary classification
        y = (iris.target == 2)  # Iris-Virginica

        # Plot Original Data Set - Virginica and Non-Virginica instances
        colors = ['green', 'blue']
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap= ListedColormap(colors))
        plt.show()

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X)

        svm_classifier = LinearSVC(C=1, loss="hinge")
        svm_classifier.fit(X_tr, y)

        print("A flower with petal length = 5.5 and petal width = 1.7 is Iris-Virginica?", svm_classifier.predict([[5.5, 1.7]]))

        # Multiclass classification?
        y = iris.target
        X_tr = scaler.fit_transform(X)

        svm_classifier = LinearSVC(C=1, loss="hinge")
        svm_classifier.fit(X_tr, y)

        print("A flower with petal length = 5.5 and petal width = 1.7 is:", iris.target_names[svm_classifier.predict([[5.5, 1.7]])[0]])
