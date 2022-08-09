import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Read in the advertising.csv file and set it to a data frame called ad_data

path = os.getcwd()
ad_data = pd.read_csv(path+'/data/advertising.csv')

# Explore the data

ad_data.head()
ad_data.info()
ad_data.describe()

# Exploratory Data Analysis

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


sns.jointplot(x='Age',y='Area Income',data=ad_data)      # A jointplot showing Area Income versus Age
plt.show()


sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,fill=True,color='red',kind='kde')     # A jointplot showing the kde distributions of Daily Time spent on site vs. Age
plt.show()


sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')  # A jointplot of Daily Time Spent on Site vs. Daily Internet Usage
plt.show()


sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')   # A pairplot with the hue defined by the 'Clicked on Ad' column feature
plt.show()

# Logistic Regression

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  # split the data into train and test data

# Train and fit a logistic regression model on the training set

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# Predictions and Evaluations

predictions = logmodel.predict(X_test)

print(classification_report(y_test,predictions))
