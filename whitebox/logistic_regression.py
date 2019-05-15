from base import ExperimentBase
from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionExperiment(ExperimentBase):
    async def do_run_async(self):
        iris = load_iris()  # This is of type Bunch - a subclass of dict that supports attribute-style access (a la JavaScript)
        print(iris.keys())
        # Last column of all arrays in first dimension - petal width
        # Note we use iris.data[:, 3:] instead of iris.data[:, 3] which is conceptually equivalent
        # in order to get a shape of (150,1) as opposed to (150,) - estimators-predictors in Scikit-Learn
        # work with bi-dimensional ndarrays for X (shape (n_samples, n_features))
        X = iris.data[:, 3:]
        # Target is a ndarray of shape (150,) containing multiclass labels: 0, 1, 2 and 3
        # Transform to binary labels: True if Iris-Virginica, False otherwise
        y = (iris.target == 2)

        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        # Let's predict on a set of 20 instances with petal width a float between 0 and 3
        # The petal width of Iris-Virginica flowers ranges from 1.4 cm to 2.5 cm
        # while the other iris flowers generally have a smaller petal width, ranging from 0.1 cm to 1.8 cm.
        X_new = np.random.uniform(0, 3, (20, 1))

        # Equivalent to:
        # X_new = []
        # for i in range(0, 20):
        #    X_new.append(np.random.uniform(0, 3))

        # X_new = np.array(X_new)
        # X_new = X_new.reshape((20, 1))
        print(f"Trained with a single feature. Logistic Regression Predicted Probabilities:{log_reg.predict_proba(X_new)}")
        print(f"Trained with a single feature. Logistic Regression Predicted Binary Labels:{log_reg.predict(X_new)}")

        # Train with all the features (sepal length, sepal width, petal length, petal width)
        X = iris.data
        log_reg = LogisticRegression()
        log_reg.fit(X, y)

        X_new = np.random.uniform(0, 5, (20, 4))

        print(f"Trained with all features. Logistic Regression Predicted Binary Labels:{log_reg.predict(X_new)}")

        # Softmax Regression (Multinomial Logistic Regression) (Generalize Logistic Regression to multiclass classification)
        X = iris.data
        y = iris.target

        softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=100)
        softmax_reg.fit(X, y)

        print(f"Softmax prediction for a flower with sepal and petal length = 5 and width = 2:{iris.target_names[softmax_reg.predict([[5, 2, 5, 2]])[0]]}")



