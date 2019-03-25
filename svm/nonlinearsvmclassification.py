from base import ExperimentBase
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
import numpy as np


class NonLinearSvmClassificationExperiment(ExperimentBase):
    async def do_run_async(self):
        # A random dataset returning 100 sample instances with 2 features and 100 binary labels (1 or 0)
        # The instances are not lineary separable like the iris dataset in order to test polynomial SVM classification
        moons = make_moons()
        X = moons[0]
        y = moons[1]

        # Plot?

        pipeline = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),  # Polynomial degree is usually number of features + 1?
            ("scaler", StandardScaler())
        ])
        X_tr = pipeline.fit_transform(X)

        classifier = LinearSVC(C=10, loss="hinge", max_iter=10000)
        classifier.fit(X_tr, y)

        random_index = np.random.randint(0, 100)
        print("Non-linear SVM prediction:", classifier.predict([X_tr[random_index, ]]))
        print("Label:", y[random_index])
