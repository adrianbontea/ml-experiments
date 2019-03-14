from base.base import ExperimentBase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegressionNormalEquationExperiment(ExperimentBase):
    async def do_run_async(self):
        # Generate some linear-looking data to test the normal equation
        # Note: X is a ndarray of shape (100,1) simulating a training set of 100 instances with one feature each
        # y is a ndarray of shape (100,1) simulating the labels for the 100 training instances
        # y is a linear-ish model introducing some noise (+ np.random.randn(100, 1)) to the linear equation (y = ax + b)
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)

        plt.plot(X, y, ".")
        plt.show()

        # Normal equation (Xt * X)pwr(-1) * Xt * y
        # The equation gives the value of Theta that minimizes the MSE cost function
        # Note the inverse of a matrix A noted A pwr(-1) is a matrix in such a way that
        # A x A pwr(-1) = I (identity matrix = 1 on the main diagonal and all other elements 0)
        # See this for details of how the inverse of a matrix is computed:
        # https://www.mathsisfun.com/algebra/matrix-inverse.html
        # In numpy this is computed by the inv() function

        # Add bias X0 = 1 to each instance
        X_b = np.append(np.ones((100, 1)), X, axis=1)
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        # Note the result is a ndarray of shape (2,1) corresponding to a single feature for the instances:
        # theta0 (bias term) and theta1 (feature weight)
        print("Parameters vector as determined by the normal equation:", theta_best)
        # Note the result is close to 4 and 3 in the original equation (y = 4 + 3 * X + np.random.randn(100, 1))
        # but not exactly 4 and 3 because of the extra noise (+ np.random.randn(100, 1))

        # Use the parameters computed via normal equation to make predictions
        X_new = np.array([[0], [2]])
        X_new_b = np.append(np.ones((2, 1)), X_new, axis=1)
        y_predict = X_new_b.dot(theta_best)

        print("Predictions based on parameters computed via normal equation:", y_predict)

        # Compute the same using Scikit-Learn and compare - the predictions should be identical!
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        print("Predictions done by Scikit-Learn:", lin_reg.predict(X_new))

        # Note: The Normal Equation gets very slow when the number of features grows large!
        # On the positive side, this equation is linear with regards to the number of instances in
        # the training set (it is O(m)), so it handles large training sets efficiently, provided they
        # can fit in memory.
        # Also, once you have trained your Linear Regression model, predictions are very fast:
        # the computational complexity is linear with regards to both the number of instances
        # you want to make predictions on and the number of features.
        # In other words, making predictions on twice as many
        # instances (or twice as many features) will just take roughly twice as much time.



