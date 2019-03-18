from base.base import ExperimentBase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class PolynomialRegressionExperiment(ExperimentBase):
    async def do_run_async(self):
        # Generate some non-linear data based on a quadratic equation
        m = 100
        X = 6 * np.random.rand(m, 1) - 3
        y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

        plt.plot(X, y, ".")
        plt.show()

        # You can use a linear model to fit non-linear data
        # by adding powers of each feature as new features to the training instances
        # and training a linear model to fit this data

        polly = PolynomialFeatures(degree=2, include_bias=False)
        X_tr = polly.fit_transform(X)

        print(X[0, ])
        print(X_tr[0, ])

        # If there were two features a and b, PolynomialFeatures with degree=3 would not only add the
        # features a**2, a**3, b**2, and b**3, but also the combinations ab, a**2b, and ab**2.

        # Let's train a linear model on the polynomial data
        reg = LinearRegression()
        reg.fit(X_tr, y)

        # Let's try a prediction
        random_index = np.random.randint(0, m)
        print("Value predicted using LinearRegression: ", reg.predict(np.array([X_tr[random_index, ]])))
        print("Label:", y[random_index, ])
