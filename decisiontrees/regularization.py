from base import ExperimentBase
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons


class DecisionTreeRegularizationExperiment(ExperimentBase):
    async def do_run_async(self):
        # A random dataset returning 100 sample instances with 2 features and 100 binary labels (1 or 0)
        # The instances are not lineary separable like the iris dataset
        moons = make_moons()
        X = moons[0]
        y = moons[1]

        # Plot Original Data Set
        colors = ['green', 'blue']
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(colors))
        plt.show()
        # Decision Trees make very few assumptions about the training data (as opposed to linear
        # models, which obviously assume that the data is linear, for example). If left
        # unconstrained, the tree structure will adapt itself to the training data, fitting it very
        # closely, and most likely overfitting it.
        # To avoid overfitting the training data, you need to restrict the Decision Tree’s freedom
        # during training. This is called regularization. The regularization
        # hyperparameters depend on the algorithm used, but generally you can at least restrict
        # the maximum depth of the Decision Tree. In Scikit-Learn, this is controlled by the
        # max_depth hyperparameter (the default value is None, which means unlimited).
        # Reducing max_depth will regularize the model and thus reduce the risk of overfitting.

        # The DecisionTreeClassifier class has a few other parameters that similarly restrict
        # the shape of the Decision Tree: min_samples_split (the minimum number of samples a node must have before it can be split),
        # min_samples_leaf (the minimum number of samples a leaf node must have),
        # max_leaf_nodes (maximum number of leaf nodes), and max_features
        # (maximum number of features that are evaluated for splitting at each node). Increasing
        # min_* hyperparameters or reducing max_* hyperparameters will regularize the
        # model.

        # Train and visualize a depth 7 Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=7)
        tree_clf.fit(X, y)

        plot_decision_regions(X, y, tree_clf)
        plt.title('Decision Tree - Depth 7 on Moons')
        plt.show()

        # Train and visualize a more regularized Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4)
        tree_clf.fit(X, y)

        plot_decision_regions(X, y, tree_clf)
        plt.title('Decision Tree - Depth 3 on Moons')
        plt.show()

