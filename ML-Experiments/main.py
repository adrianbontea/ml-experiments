import californiahousing.helpers as hlp
import matplotlib.pyplot as plt

data = hlp.load_housing_data()

# Analize the housing data
print("Quick peek at top 5 instances:")
print(data.head())
input("Hit ENTER to continue")

print("Ocean Proximity counts in categories:")
print(data.ocean_proximity.value_counts())
input("Hit ENTER to continue")

print("Description of the data set (percentiles etc):")
print(data.describe())
input("Hit ENTER to continue")

print("Slice a smaller DataFrame and plot:")
data[["median_house_value","median_income"]].hist()
plt.show()

print("Slice first 3 rows:")
print(data[0:3])
input("Hit ENTER to continue")

print("Slice first 3 rows and 5 columns:")
print(data.iloc[0:3,0:5])
input("Hit ENTER to continue")

print("Query: Instances with Median Income == 10:")
print(data[data.median_income == 10])
input("Hit ENTER to continue")

print("Query: Instances with Median Income > 10:")
print(data[data.median_income > 10])
input("Hit ENTER to continue")


