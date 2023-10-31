# Market
# my_project_phases
Dataset link: Kaggle, " Assignment-1_Data.csv”, Dataset Owner: Aslan Ahmedov, MBA DS & AI at Ascencia Business School https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis 

#PREPROCESSING AND VISUALIZATION

#Step: 1 Import the necessary library and import the the association rules.

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori from mlxtend.frequent_patterns

#Step: 2 Upload the dataset

from google.colab import files
data = files.upload() 
Saving Assignment-1_Data.xlsx to Assignment-1_Data.xlsx

df = pd.read_excel('Assignment-1_Data.xlsx')
print(df)

#Step: 3 Display the particular rows and columns in a dataset.

df.head(20)

df = retail_data_prep(df)
df.describe().T 
df.isnull().sum()

#Step: 4 Display the frequency items from the dataset using association rules.

import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data = pd.read_excel('Assignment-1_Data.xlsx') 
transactions = data.groupby(['Price', 'Itemname'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Price') 
transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)
min_support = 0.05 
frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
min_threshold = 1.0 
association_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold) print("Frequent Itemsets:")
print(frequent_itemsets) print("\nAssociation Rules:") print(association_rules)

#Visualization: 

import seaborn as sns
dataset = pd.read_excel("Assignment-1_Data.xlsx") 
print(dataset.head()) 
plt.figure(figsize=(8, 6)) 
sns.histplot(dataset, kde=True)
plt.xlabel('X-axis label') 
plt.ylabel('Y-axis label') 
plt.title('Histogram of Numerical Column') 
plt.show() plt.figure(figsize=(8, 6)) 
sns.scatterplot(x='Itemname', y='Price', data=dataset)
plt.xlabel('X-axis label') plt.ylabel('Y-axis label') 
plt.title('Scatter Plot of Column1 vs. Column2') 
plt.figure(figsize=(8, 6)) sns.countplot(x='Itemname', data=dataset) 
plt.xlabel('Quantity') plt.ylabel('Price') 
plt.title('Bar Plot of Categorical Column')
plt.xticks(rotation=45)
plt.show()

Import the decision tree for training and testing the dataset

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

#Load the data (replace 'data.csv' with your data file)
df = pd.read_excel('Assignment-1_Data.xlsx')

#Data cleaning and handling missing values (if any)
df.dropna(inplace=True)

#You can use other methods to handle missing values, like imputation.
#Split the data into features (X) and target (y)
X = df.drop('Itemname', axis=1) # Replace 'target_column' with the name of your target variable
y = df['Itemname'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(X,y)

#Data cleaning and handling missing values (if any)
df.dropna(inplace=True)

#Extract target variable
y = df['Itemname']

#Drop non-numeric columns (you may need to adjust this based on your data)
X = df.select_dtypes(include=['number'])

#One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

#Train the model
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred) conf_matrix = confusion_matrix(y_test, y_pred) class_report = classification_report(y_test, y_pred)

#Print the results

print(f"Accuracy: {accuracy}") 
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
