import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MinMaxScaler
from .base import HousingExperimentBase


class PreProcessingExperiment(HousingExperimentBase):
    async def do_run_async(self):
        data = super().load_housing_data()
        train_set, test_set = super().get_train_test(data, 0.2)

        predictors = train_set.drop("median_house_value", axis=1)

        # Fill total_bedrooms in train set and test set with the median value where null
        median = predictors.total_bedrooms.median()
        predictors.total_bedrooms.fillna(median)
        test_set.total_bedrooms.fillna(median)

        # Equivalent to next using Scikit-Learn Imputer but this one with strategy="median" works across all numeric
        # columns
        imputer = SimpleImputer(strategy="median")
        # Only numeric values
        predictors_num = predictors.drop("ocean_proximity", axis=1)
        # Learn the medians and transform - this returns an 2-dimensional NumPy array (ndarray) of shape (16512,
        # 8) (16512 rows of instances with 8 columns) including the transformed features

        # Scikit-Learn estimators/transformers/predictors work with 2 dimensional matrix-like data structures for X data set
        # of shape (n_samples, n_features). Both a Pandas DataFrame and numpy ndarray work
        # The _get_values method of a DataFrame returns the corresponding ndarray
        x = imputer.fit_transform(predictors_num)
        print(type(x).__name__)
        print(x.shape)

        # Put it back into a DataFrame
        predictors_tr = pd.DataFrame(x, columns=predictors_num.columns)

        # Convert ocean_proximity text values to numbers
        encoder = LabelEncoder()
        ocean_proximity_encoded = encoder.fit_transform(predictors.ocean_proximity)
        predictors_tr["ocean_proximity"] = ocean_proximity_encoded

        # Problem with this approach is that ML algorithms will consider 2 close values are more similar than 2
        # distant values which might not be the case This can be fixed using one-hot encoding for text values (the
        # distinct values are like classes)

        binarizer = LabelBinarizer()
        # This returns a 2-dimensional NumPy array (ndarray) of shape (16512,5) (16512 binary arrays of size 5 - a
        # single 1 and rest 0 to denote a certain class)
        ocean_proximity_1_hot = binarizer.fit_transform(predictors.ocean_proximity)
        print(type(ocean_proximity_1_hot).__name__)
        print(ocean_proximity_1_hot.shape)

        # Transform the numeric predictors using a pipeline composed of an inputer and scaler
        pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), ('scaler', MinMaxScaler())])
        predictors_tr = pipeline.fit_transform(predictors_num)

        # predictors_tr is now a 2 dimensional nd array (16512, 8) - convert it back to a data frame
        predictors_tr = pd.DataFrame(predictors_tr, columns=predictors_num.columns)

        # ocean proximity one hot vector is now a 2 dimensional ndarray of shape (16512,5) - convert it to list in
        # order to be able to save it as a data frame column.
        # The result is a python list of lists (a list containing 16512 lists each containing 5 binary values).
        # The final data frame will contain a list of 5 binary values in the ocean_proximity column for each row
        ocean_proximity = ocean_proximity_1_hot.tolist()
        predictors_tr["ocean_proximity"] = ocean_proximity
        print(predictors_tr.head())
