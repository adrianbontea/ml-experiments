from base.base import ExperimentBase
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegressionExperiment(ExperimentBase):
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

        # Fit the training set (determine theta best) using Batch Gradient Descent.
        # The result should be the same as normal equation
        # Batch gradient descent involves computing the derivative of the MSE cost function
        # with respect to each parameter from the parameters vector at each training step!
        # That is because the derivative of a function determines the slope of the tangent
        # to the function curve in a certain point. Hence these partial derivatives are about determining
        # the slope of the cost function with regards to each axis
        # represented by each model parameter, trying to reach a global minimum for the cost function.
        # For a certain parameter theta j, the derivative (gradient) will be: 2/m * Sum i=1 -> m(Theta T * xi - yi)*xi,j (feature j from instance i)
        # A vector of all these gradients for the whole training set: 2/m * X T *(X * Theta - y)

        # This is why the algorithm is called Batch Gradient Descent: it uses the whole batch of training
        # data at every step. As a result it is terribly slow on very large training
        # sets. However, Gradient Descent scales well with the number of
        # features; training a Linear Regression model when there are hundreds of thousands of features
        # is much faster using Gradient Descent than using the Normal Equation.

        eta = 0.1  # learning rate
        n_iterations = 1000
        m = 100

        theta = np.random.randn(2, 1)  # random initialization

        for iteration in range(n_iterations):
            gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
            theta = theta - eta * gradients

        print("Parameters vector as determined by Batch Gradient Descent:", theta)
        # These should be precisely the values determined by the normal equation!
        # Note: if theta best is found before number of steps is hit (1000 in this case)
        # theta next should be equal to theta since the gradients vector will be all 0!
        # remember that the derivative of a function determines the slope of the tangent in a certain point
        # so for a global minimum point the inclination should be 0!

        # Batch Gradient Descent is very slow when the training set is large as it uses the whole set on each iteration
        # Stochastic (Random) Gradient Descent is a similar algorithm but uses a single instance,
        # randomly chosen from the training set, on each iteration, to compute the gradients vector and theta next.
        # It is much faster but less accurate. To avoid bouncing around after finding the best parameters vector that minimizes
        # the MSE function, the eta (learning rate) is gradually reduced in a process called learning schedule.
        # This process is called simulated annealing, because it resembles the process of annealing in metallurgy
        # where molten metal is slowly cooled down.

        # By convention we iterate by rounds of m iterations; each round is called an epoch.
        # While the Batch Gradient Descent code iterated 1,000 times through the whole training
        # set, this code goes through the training set only 50 times and reaches a fairly good solution

        n_epochs = 50

        def learning_schedule(t):  # Local function
            t0, t1 = 5, 50
            return t0 / (t + t1)

        theta = np.random.randn(2, 1)  # random initialization

        for epoch in range(n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = np.array([X_b[random_index]])
                yi = np.array([y[random_index]])
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta = learning_schedule(epoch * m + i)  # This will gradually reduce eta on each epoch as i increases and globally as epoch increases
                theta = theta - eta * gradients

        print("Parameters vector as determined by Stochastic Gradient Descent:", theta)


