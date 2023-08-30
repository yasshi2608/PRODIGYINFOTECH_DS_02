import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)
# Display basic information about the dataset
print(titanic_df.info())

# Display summary statistics
print(titanic_df.describe())

# Display the first few rows of the dataset
print(titanic_df.head())
# Check for missing values
print(titanic_df.isnull().sum())

# Remove rows with missing values or impute them
titanic_df.dropna(subset=['Age', 'Embarked'], inplace=True)
titanic_df['Cabin'].fillna('Unknown', inplace=True)

# Drop unnecessary columns
titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# Distribution of passenger ages
plt.figure(figsize=(8, 6))
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Survival rate by sex
sns.catplot(x='Sex', hue='Survived', data=titanic_df, kind='count')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Relationship between fare and age
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', data=titanic_df, hue='Survived')
plt.title('Relationship between Fare and Age')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Correlation matrix
correlation_matrix = titanic_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()





