from base.experimentbase import ExperimentBase
import californiahousing.helpers as hlp
import matplotlib.pyplot as plt

class DataAnalysisExperiment(ExperimentBase):
    def run(self):    
        data = hlp.load_housing_data()

        # Analize the housing data
        print("Quick peek at top 5 instances:")
        print(data.head())

        print("Ocean Proximity counts in categories:")
        print(data.ocean_proximity.value_counts())

        print("Description of the data set (percentiles etc):")
        print(data.describe())

        print("Slice a smaller DataFrame and plot:")
        data[["median_house_value","median_income"]].hist()
        plt.show()

        print("Slice first 3 rows:")
        print(data[0:3])

        print("Slice first 3 rows and 5 columns:")
        print(data.iloc[0:3,0:5])

        print("Query: Instances with Median Income == 10:")
        print(data[data.median_income == 10])

        print("Query: Instances with Median Income > 10:")
        print(data[data.median_income > 10])

        train_set, test_set = hlp.get_train_test(data, 0.2)
        print("Train set size is " + str(len(train_set)))
        print("Test set size is " + str(len(test_set)))

        train_set_copy = train_set.copy()


