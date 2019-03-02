from californiahousing.base import HousingExperimentBase
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class PreProcessingExperiment(HousingExperimentBase): 
    async def do_run_async(self):
        data = super().load_housing_data()
        train_set, test_set = super().get_train_test(data, 0.2)

        predictors = train_set.drop("median_house_value", axis = 1)
        labels = train_set.median_house_value.copy()

        # Fill total_bedrooms in train set and test set with the median value where null
        median = predictors.total_bedrooms.median()
        predictors.total_bedrooms.fillna(median)
        test_set.total_bedrooms.fillna(median)

        # Equivalent to next using Scikit-Learn Imputer but this one with strategy="median" works across all numeric columns
        imputer = SimpleImputer(strategy="median")
        # Only numeric values
        predictors_num = predictors.drop("ocean_proximity", axis = 1)
        # Learn the medians and transform - this returns an 2-dimensional NumPy array (ndarray) of shape (16512,8) (16512 rows of instances with 8 columns) including the transformed features
        X = imputer.fit_transform(predictors_num)
        print(type(X).__name__)
        print(X.shape)

        # Put it back into a DataFrame
        predictors_tr = pd.DataFrame(X, columns=predictors_num.columns)

        # Convert ocean_proximity text values to numbers
        encoder = LabelEncoder()
        ocean_proximity_encoded = encoder.fit_transform(predictors.ocean_proximity)
        predictors_tr["ocean_proximity"] = ocean_proximity_encoded
        
        # Problem with this approach is that ML algorithms will consider 2 close values are more similar than 2 distant values which might not be the case
        # This can be fixed using one-hot encoding for text values (the distinct values are like classes)

        binarizer = LabelBinarizer()
        # This returns a 2-dimensional NumPy array (ndarray) of shape (16512,5) (16512 binary arrays of size 5 - a single 1 and rest 0 to denote a certain class)
        ocean_proximity_1_hot = binarizer.fit_transform(predictors.ocean_proximity)
        print(type(ocean_proximity_1_hot).__name__)
        print(ocean_proximity_1_hot.shape)

        # Transform the numeric predictors using a pipeline composed of an inputer and scaler
        pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('scaler', MinMaxScaler())])
        predictors_tr = pipeline.fit_transform(predictors_num)

        #predictors_tr is now a 2 dimensional nd array (16512, 8) - convert it back to a data frame
        predictors_tr = pd.DataFrame(predictors_tr, columns=predictors_num.columns)

        # ocean proximity one hot vector is now a 2 dimensional ndarray of shape (16512,5) - convert it to list in order to be able to save it as a data frame column
        predictors_tr["ocean_proximity"] = ocean_proximity_1_hot.tolist()
        print(predictors_tr.head())

        # Finally a full pipeline to transform both the numeric predictors and ocean_proximity into a single data set
        # Note: This doesn't work yet
        #num_attribs = list(predictors_num) # All column names from the numerical predictors DataFrame
        #cat_attribs = ["ocean_proximity"]

        #num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', SimpleImputer(strategy="median")),('scaler', MinMaxScaler())])
        #cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),('encoder', LabelBinarizer())])

        #full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline)])
        #predictors_prepared = full_pipeline.fit_transform(predictors)

        #print(predictors_prepared.shape)