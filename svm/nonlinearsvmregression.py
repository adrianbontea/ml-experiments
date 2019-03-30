from base import ExperimentBase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import PolynomialFeatures


class NonLinearSvmRegressionExperiment(ExperimentBase):
    async def do_run_async(self):
        # Generate some non-linear data based on a quadratic equation
        m = 100
        X = 6 * np.random.uniform(1, 5, (m, 1)) - 3
        y = 0.5 * X ** 2 + X + 2 + np.random.uniform(1, 5, (m, 1))

        plt.plot(X, y, ".")
        plt.show()

        # To tackle nonlinear regression tasks, you can use a kernelized SVM model
        svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
        svm_poly_reg.fit(X, y)

        rand_index = np.random.randint(0, 99)
        x = X[rand_index, ]
        print("Prediction for:", x)
        print(svm_poly_reg.predict([x]))
        print("Label:", y[rand_index, ])

        # ... or just use the Linear SVR algorithm with polynomial features
        polly = PolynomialFeatures(degree=2)  # Polynomial degree is usually number of features + 1?
        X_tr = polly.fit_transform(X)

        svm_reg = LinearSVR(epsilon=1.5)
        svm_reg.fit(X_tr, y)

        rand_index = np.random.randint(0, 99)
        x = X_tr[rand_index, ]
        print("Prediction for:", x)
        print(svm_reg.predict([x]))
        print("Label:", y[rand_index, ])
