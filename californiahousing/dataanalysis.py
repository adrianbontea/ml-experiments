from .base import HousingExperimentBase
import matplotlib.pyplot as plt


class DataAnalysisExperiment(HousingExperimentBase):
    async def do_run_async(self):
        data = super().load_housing_data()

        # Analyze the housing data
        print("Quick peek at top 5 instances (districts):")
        print(data.head())

        print("Ocean Proximity counts in categories:")
        print(data.ocean_proximity.value_counts())

        print("Description of the data set (percentiles etc):")
        print(data.describe())

        print("Slice a smaller DataFrame and plot:")
        subset = data[["median_house_value", "median_income"]]
        subset.hist()

        subset = data["median_house_value"]
        print(type(subset).__name__)  # Series

        subset = data[["median_house_value"]]
        print(type(subset).__name__)  # DataFrame

        print("Slice first 3 rows:")
        subset = data[0:3]
        print(subset)

        print("Slice first 3 rows and 5 columns:")
        subset = data.iloc[0:3, 0:5]
        print(subset)

        print("Query: Districts with Median Income == 10:")
        subset = data[data.median_income == 10]
        print(subset)

        print("Query: Districts with Median Income > 10:")
        subset = data[data.median_income > 10]
        print(subset)

        train_set, test_set = super().get_train_test(data, 0.2)
        print("Train set size is " + str(len(train_set)))
        print("Test set size is " + str(len(test_set)))

        train_set_copy = train_set.copy()

        train_set_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

        train_set_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                            s=train_set_copy.population / 100, label="population",
                            c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

        plt.legend()
        plt.show()

        correlation_matrix = train_set_copy.corr()
        print(correlation_matrix["median_house_value"].sort_values(ascending=False))

        # The best correlation for predicting the median housing value seems to be with the median income attribute
        # so plot their relation
        train_set_copy.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
        plt.show()

        train_set_copy["rooms_per_household"] = train_set_copy["total_rooms"] / train_set_copy["households"]
        correlation_matrix = train_set_copy.corr()
        print(correlation_matrix["median_house_value"].sort_values(ascending=False))
