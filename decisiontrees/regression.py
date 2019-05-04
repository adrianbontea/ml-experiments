from base import ExperimentBase
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import os
import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeRegressionExperiment(ExperimentBase):
    async def do_run_async(self):
        # Generate some non-linear data based on a quadratic equation
        m = 100
        X = 6 * np.random.uniform(1, 5, (m, 1)) - 3
        y = 0.5 * X ** 2 + X + 2 + np.random.uniform(1, 5, (m, 1))

        plt.plot(X, y, ".")
        plt.show()

        tree_reg = DecisionTreeRegressor(max_depth=2)
        tree_reg.fit(X, y)

        output_folder = "output/decisiontrees"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        export_graphviz(
            tree_reg,
            out_file=os.path.join(output_folder, "tree_regressor_depth2.dot"),
            rounded=True,
            filled=True
        )

        rand_index = np.random.randint(0, 99)
        x = X[rand_index, ]
        print(f"Prediction for:{x}")
        print(tree_reg.predict([x]))
        print(f"Label:{y[rand_index, ]}")
