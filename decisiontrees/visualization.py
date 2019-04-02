from base import ExperimentBase
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class DecisionTreeVisualizationExperiment(ExperimentBase):
    async def do_run_async(self):
        iris = load_iris()
        X = iris.data[:, 2:]  # petal length and width
        y = iris.target

        output_folder = "output/decisiontrees"
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Train and visualize a depth 2 Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=2)
        tree_clf.fit(X, y)

        export_graphviz(
            tree_clf,
            out_file=os.path.join(output_folder, "iris_tree_depth_2.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

        # Export .dot file to PNG with a command like (on Windows):
        # "C:\Program Files (x86)\Graphviz2.38\bin\dot.exe" -Tpng .\output\decisiontrees\iris_tree.dot -o .\output\decisiontrees\iris_tree.png

        # A node’s samples attribute counts how many training instances it applies to. For
        # example, 100 training instances have a petal length greater than 2.45 cm (depth 1,
        # right), among which 54 have a petal width smaller than 1.75 cm (depth 2, left). A
        # node’s value attribute tells you how many training instances of each class this node
        # applies to: for example, the bottom-right node applies to 0 Iris-Setosa, 1 Iris-
        # Versicolor, and 45 Iris-Virginica. Finally, a node’s gini attribute measures its impurity:
        # a node is “pure” (gini=0) if all training instances it applies to belong to the same
        # class.
        # Impurity formula for node i: Gi = 1 - Sum k = 1 -> n (Pi,k)**2
        # n = number of classes
        # Pi,k is the ratio of class k instances among the training instances to which node i applies
        # e.g. For node at depth 2 left P3,1 = 49/54

        plot_decision_regions(X, y, tree_clf)
        plt.xlabel('Petal Length [cm]')
        plt.ylabel('Petal Width [cm]')
        plt.title('Decision Tree - Depth 2 on Iris - 2 Features - Multi Classes')
        plt.show()

        # Train and visualize a depth 3 Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=3)
        tree_clf.fit(X, y)

        export_graphviz(
            tree_clf,
            out_file=os.path.join(output_folder, "iris_tree_depth_3.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

        plot_decision_regions(X, y, tree_clf)
        plt.xlabel('Petal Length [cm]')
        plt.ylabel('Petal Width [cm]')
        plt.title('Decision Tree - Depth 3 on Iris - 2 Features - Multi Classes')
        plt.show()

        # Train and visualize a depth 7 Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=7)
        tree_clf.fit(X, y)

        export_graphviz(
            tree_clf,
            out_file=os.path.join(output_folder, "iris_tree_depth_7.dot"),
            feature_names=iris.feature_names[2:],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

        plot_decision_regions(X, y, tree_clf)
        plt.xlabel('Petal Length [cm]')
        plt.ylabel('Petal Width [cm]')
        plt.title('Decision Tree - Depth 7 on Iris - 2 Features - Multi Classes')
        plt.show()

        # From the decision regions chart and from the graphical representation of the decision tree
        # it becomes obvious that the model increases accuracy with the depth:
        # The decision regions capture instances more accurate (a more complex model)
        # while gini is 0 for all bottom leaf nodes of the tree in the graphical representation...

        # Finally let's try a Decision Tree Classifier trained with all 4 features and depth of 7
        X = iris.data
        y = iris.target

        tree_clf = DecisionTreeClassifier(max_depth=7)
        tree_clf.fit(X, y)

        export_graphviz(
            tree_clf,
            out_file=os.path.join(output_folder, "iris_4_features_tree_depth_7.dot"),
            feature_names=iris.feature_names,
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )

        # Cannot plot decision regions because of multiple feature instances
        # (multi-dimensional space for feature vectors...)
