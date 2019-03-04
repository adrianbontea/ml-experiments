import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from californiahousing.base import HousingExperimentBase


class ModelsExperiment(HousingExperimentBase):
    async def do_run_async(self):
        # Load, split and pre-process the data (part of the bigger pre-processing experiment)
        data = super().load_housing_data()
        train_set, test_set = super().get_train_test(data, 0.2)

        predictors = train_set.drop("median_house_value", axis=1)
        labels = train_set.median_house_value.copy()

        predictors_tr = self.__transform(predictors)

        # Linear Regression
        lin_reg = LinearRegression()
        # Learn
        lin_reg.fit(predictors_tr, labels)

        # Test with first 5 instances from the training set
        some_data = predictors_tr[:5]
        some_labels = labels[:5]

        print("Linear Regression Predictions:", lin_reg.predict(some_data))
        print("Labels:", list(some_labels))

        # Measure the error using RMSE function
        predictions = lin_reg.predict(predictors_tr)
        lin_mse = mean_squared_error(labels, predictions)
        lin_rmse = np.sqrt(lin_mse)

        print("Linear Regression RMSE for the whole training set is:", lin_rmse)

        # 68434 is not great at all! (basically means a typical prediction error of $68434). The model is
        # underfitting (too simple)

        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(predictors_tr, labels)

        print("Decision Tree Predictions:", tree_reg.predict(some_data))
        print("Labels:", list(some_labels))

        # Measure the error using RMSE function
        predictions = tree_reg.predict(predictors_tr)
        lin_mse = mean_squared_error(labels, predictions)
        lin_rmse = np.sqrt(lin_mse)

        print("Decision Tree RMSE for the whole training set is:", lin_rmse)

        # Decision Tree looks perfect (0 RMSE) but it's not! It just performs perfect on the training set that it
        # learned but won't generalize well to new data as demonstrated below(overfitting) Split the training set in
        # 2, train using the first part and verify using the other

        train_set2, test_set2 = super().get_train_test(train_set, 0.2)
        predictors2 = train_set2.drop("median_house_value", axis=1)
        labels2 = train_set2.median_house_value.copy()

        predictors_tr2 = self.__transform(predictors2)
        tree_reg.fit(predictors_tr2, labels2)

        predictors2 = test_set2.drop("median_house_value", axis=1)
        labels2 = test_set2.median_house_value.copy()

        predictors_tr2 = self.__transform(predictors2)

        test_data = predictors_tr2[:5]
        test_labels = labels2[:5]

        print("Decision Tree Predictions:", tree_reg.predict(test_data))
        print("Labels:", list(test_labels))

        # Perform K-fold cross-validation of Decision Tree model to measure the RMSE
        scores = cross_val_score(tree_reg, predictors_tr, labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)

        print("Decision Tree Scores:", rmse_scores)
        print("Decision Tree Mean:", rmse_scores.mean())
        print("Decision Tree Standard deviation:", rmse_scores.std())

        # Train and test a Random Forest Regressor (works by training multiple Decision Trees on different subsets of
        # the training set and averaging based on their predictions)
        forest_reg = RandomForestRegressor()
        forest_reg.fit(predictors_tr, labels)

        print("Random Forest Predictions:", forest_reg.predict(some_data))
        print("Labels:", list(some_labels))

        # Measure the error using RMSE function
        predictions = forest_reg.predict(predictors_tr)
        lin_mse = mean_squared_error(labels, predictions)
        lin_rmse = np.sqrt(lin_mse)

        print("Random Forest RMSE for the whole trainig set is:", lin_rmse)

        # Finally, let's test the Random Forest (most promising from the 3 models although still overfiting quite a
        # lot) on the test set!
        predictors = test_set.drop("median_house_value", axis=1)
        labels = test_set.median_house_value.copy()

        predictors_tr = self.__transform(predictors)

        test_data = predictors_tr[10:30:2]
        test_labels = labels[10:30:2]

        print("Decision Tree Predictions:", tree_reg.predict(test_data))
        print("Labels:", list(test_labels))

    def __transform(self, data):
        predictors_num = data.drop("ocean_proximity", axis=1)

        binarizer = LabelBinarizer()
        # This returns a 2-dimensional NumPy array (ndarray) of shape (16512,5) (16512 binary arrays of size 5 - a
        # single 1 and rest 0 to denote a certain class)
        ocean_proximity_1_hot = binarizer.fit_transform(data.ocean_proximity)
        # Transform the numeric predictors using a pipeline composed of an inputer and scaler
        pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', MinMaxScaler())])
        predictors_tr = pipeline.fit_transform(predictors_num)

        # predictors_tr is now a 2 dimensional nd array (16512,8) - Need to merge it with the binarized labels into (
        # 16512,13) shape ocean proximity one hot vector is a 2 dimensional ndarray of shape (16512,5)
        predictors_tr = np.append(predictors_tr, ocean_proximity_1_hot, axis=1)

        return predictors_tr
