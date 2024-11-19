
#%%
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

#%% Load the data

drugs_data = pd.read_csv(r'D:\Datavs\drug200 - drug200.csv.csv')

#%% Basic data summaries
# Get the head, tail, and describe summaries
head_data = drugs_data.head()
tail_data = drugs_data.tail()
describe_data = drugs_data.describe()
object_data = drugs_data.describe(include='object')

#%% Print summaries
print("Head Data:\n", head_data)
print("\nTail Data:\n", tail_data)
print("\nDescribe Data:\n", describe_data)
print("\nObject Data:\n", object_data)

# Print data types
dtypes_df = drugs_data.dtypes.reset_index()
dtypes_df.columns = ['Column', 'Data Type']
print("\nData Types:\n", dtypes_df)


# Print value counts for categorical features
print("\nSex Value Counts:\n", drugs_data['Sex'].value_counts())
print("\nBP Value Counts:\n", drugs_data['BP'].value_counts())
print("\nCholesterol Value Counts:\n", drugs_data['Cholesterol'].value_counts())
print("\nDrug Value Counts:\n", drugs_data['Drug'].value_counts())


#%% Check for missing values
# Display the count of null values for each column
print("\nCheck for Missing Values:\n", drugs_data.isnull().sum())


#%% Print counts of numerical and categorical features
numerical_features = drugs_data.select_dtypes(include=['int64', 'float64'])
categorical_features = drugs_data.select_dtypes(include=['object'])
print("\nCount of Numerical and Categorical Features")
print(f'Numerical Features Count: {numerical_features.shape[1]}')
print(f'Categorical Features Count: {categorical_features.shape[1]}')



#%% Outlier detection function and outlier count
def count_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return len(outliers)

#%% Count outliers in numerical columns
numerical_df = drugs_data.select_dtypes(include=['number'])
outlier_counts = numerical_df.apply(count_outliers)
#%% Sort outliers by count in descending order
sorted_outliers = outlier_counts.sort_values(ascending=False)
print("\nOutlier counts per numerical feature:\n", sorted_outliers)

# %%

plt.figure(figsize=(20, 120))
for n, feature in enumerate(categorical_features):
    plt.subplot(22, 2, n + 1)
    sns.countplot(x=feature, data=drugs_data, palette='viridis')
    plt.title(f'Count Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
plt.show()
# %%
for col in numerical_features:
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(drugs_data[col], kde = True)
    plt.subplot(1, 2, 2)
    sns.scatterplot(data = drugs_data, x = col, y='Drug')
