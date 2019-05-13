from base import ExperimentBase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


class LinearSvmRegressionExperiment(ExperimentBase):
    async def do_run_async(self):
        # Generate some linear-looking data
        # Note: X is a ndarray of shape (100,1) simulating a training set of 100 instances with one feature each
        # y is a ndarray of shape (100,1) simulating the labels for the 100 training instances
        # y is a linear-ish model introducing some noise (+ np.random.randn(100, 1)) to the linear equation (y = ax + b)
        X = 2 * np.random.uniform(1, 5, (100, 1))
        y = 4 + 3 * X + np.random.uniform(1, 5, (100, 1))

        plt.plot(X, y, ".")
        plt.show()

        svm_reg = LinearSVR(epsilon=1.5)
        svm_reg.fit(X, y)

        rand_index = np.random.randint(0, 99)
        x = X[rand_index, ]
        print("Prediction for:", x)
        print(svm_reg.predict([x]))
        print("Label:", y[rand_index, ])
