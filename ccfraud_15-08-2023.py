# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:34:33 2023

@author: thad
"""
'''
###############################################################################
Introduction: - Data Set Information
###############################################################################
The following data set on credit card fraud was obtained from Kaggle with the
below summary of the data set and it's features:

Context

It is important that credit card companies are able to recognize fraudulent 
credit card transactions so that customers are not charged for items that they 
did not purchase.


Content


The dataset contains transactions made by credit cards in September 2013 by 
European cardholders.
This dataset presents transactions that occurred in two days, where we have 
492 frauds out of 284,807 transactions. The dataset is highly unbalanced, 
the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a 
PCA transformation. Unfortunately, due to confidentiality issues, we cannot 
provide the original features and more background information about the data. 

Features V1, V2, … V28 are the principal components obtained with PCA.
 
The only features which have not been transformed with PCA are 'Time' and 
'Amount'.  

'Time' - contains the seconds elapsed between each transaction 
and the first transaction in the dataset. The feature 

'Amount' is the transaction Amount, this feature can be used for 
example-dependant cost-sensitive learning. 

'Class' is the response variable and it takes value 1 in case of fraud and 0 
otherwise.


Given the class imbalance ratio, we recommend measuring the accuracy using the 
Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is 
not meaningful for unbalanced classification.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import pylab as py
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.svm import OneClassSVM


# File path for credit card data set
cc_path = (r"C:/Users/lee/Desktop/study/Python Portfolio/CCFraud/creditcard.csv")

# Load csv file as data frame
ccfraud_df = pd.read_csv(cc_path, header = 0, sep = ',')

# Convert colum names to lower case.
ccfraud_df = ccfraud_df.rename(columns=str.lower)





'''
###############################################################################
Exploratory Data Analysis
###############################################################################
No missing values apparent in features when we use .info() function.

# Data type for V1, V2 to V28 is appropriately float64
# Data type for Time is appropriately float64
# Data type for Amount is appropriately float64
# Data type for Class has been changed from int64 to boolean

# Values for time are not unique. However they are in chronological order.

Therefore, to avoid bias we can just shuffle the rows in the code below.

We can explore and analyse the data with the following functions

'''

print(ccfraud_df.columns) # Prints name of each column


print('Summary of Column names, count of non-null values and data types\n', ccfraud_df.info()) # Prints column name. non-null values and data type

print('The shape of ccfraud_df (Rows, Columns) is \n', ccfraud_df.shape) # Provides shape of data set

print('The data types for each column are \n', ccfraud_df.dtypes) 

print('Preview of first 5 rows of the data set are: \n', ccfraud_df.head()) # Prints class of each column







# Checking uniqueness and unexpected values in class column
print(ccfraud_df['class'].unique()) # Prints unique values of Class variable.
# 0 is for legit transactions, 1 is for fraud transactions


# Check for imbalance in data set
legit = len(ccfraud_df.loc[ccfraud_df['class'] == 0])
fraud = len(ccfraud_df.loc[ccfraud_df['class'] == 1])

# Summary statistics
ccfraud_df.describe()


print(legit) # Returns 284,315 legit transactions
print(fraud) # Returns 492 fraud transactions

perc_fraud_trans = fraud/(fraud + legit) * 100
print('The percentage of fraud transactions is:', perc_fraud_trans,'%') # fraud transactions make up 0.1727% of total transactions


# Change data type for Class to boolean
ccfraud_df['class'] = ccfraud_df['class'].astype('bool')

# second data type check
ccfraud_df.dtypes
ccfraud_df['class'].dtype # Class data type is now appropriately changed to bool


# Randomise rows to avoid bias due to order of "Time" Column
ccfraud_df = ccfraud_df.sample(frac=1).reset_index(drop=True)



print(ccfraud_df['class'].value_counts())


'''
###############################################################################
Relationships Between Variables
###############################################################################
Columns V1 to V28 are the result of PCA transformation as provided in the
available dataset.

From the heatmap below, given that the data is incredibly imbalanced, and that not a little is known of the variables that have
Undergone PCA transformation. It makes it hard to discern any meaningful relationships in the data.

For columns "time" and "amount" which have not undergone any transaction, we also plot histograms of 
Frequency for "amount" and "time". 

We also create a scatter plot of amounts over time. Again, these plots do not show us any meaningful
relationship of these columns to fraud.



'''
# Heatmap of variables
sns.heatmap(data = ccfraud_df.corr(),cmap='coolwarm_r', annot_kws={'size':20})
plt.title('Correlation Matrix')
plt.show()
# Class distribution of legit and fraud transactions
colors = ["blue", "red"]

sns.countplot(data = ccfraud_df, x = 'class', palette=colors)
plt.title('class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()


# Distribution of Transaction Amount and Time
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = ccfraud_df['amount'].values
time_val = ccfraud_df['time'].values

sns.histplot(amount_val, ax=ax[0], color='r',  bins = 20, kde = True)
ax[0].set_title('Frequency of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.histplot(time_val, ax=ax[1], color='b', bins = 20, kde = True)
ax[1].set_title('Frequency of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()

# Scatter plot of amount vs time.
plt.scatter(ccfraud_df['time'], ccfraud_df['amount'], c='blue', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Amounts Over Time')
plt.show()

sns.boxplot(data = ccfraud_df, x = "class", y = "amount")
plt.title('Boxplot of Legit and Fraud Transaction Amounts')
plt.show()
'''
The boxplot above compares the amounts for legitimate and fraudulent transactions. We can see in our
dataset that amounts for fraudulent transactions are generally smaller. Perhaps this is indicative
of the ability for banks to monitor and flag transactions which are "out of character" for customers.

'''

'''
###############################################################################
splitting Data
###############################################################################
Below we split the data into training and test datasets.
Training set size is 801%. Test size is 20%.

'''


X = ccfraud_df.drop('class', axis=1)
y = ccfraud_df['class']

sss = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state=0)



for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    sss_train = ccfraud_df.iloc[train_index].reset_index(drop = True)
    sss_test = ccfraud_df.iloc[test_index].reset_index(drop = True)


X_train = sss_train.drop(['class'], axis = 1)
y_train = sss_train['class'].values
X_test = sss_test.drop(['class'], axis = 1)
y_test = sss_test['class'].values



'''
###############################################################################
Checks for NAN values and outliers
###############################################################################
'''

# Function that checks each collumn for NAN values
def na_check(df):
    for column in df:
        count_nan = df[column].isnull().sum()
        print('Number of missing value for column', 
              (column), ";", count_nan)

# Returns NAN values in sss_train and sss_split
na_check(sss_train)
na_check(sss_test)

#old 
# Checks entire data frame for NAN values. returns boolean
#ccfraud_df.isnull().values.any() # Returns false

# Provides a count of the number of NAN values in a dataframe.
#ccfraud_df.isnull().sum().sum() # Returns number of NAN values

'''
###############################################################################
Univariate Outliers
###############################################################################
Outliers are observations that lies an abnormal distance from other values.
They can occur in a number of ways:
- Outliers can be due to measurement error.
- Incorrect data entry
- Or they can represent true values from natural variation in the population.


We can use a number of methods to detect and handle outliers. We can use box plots to find them.

Use plots to identify outliers in columns. With the exception for "time", we can see that outliers 
exist in every column.

Given that it is normal practice to treat outliers prior to performing transformations such as 
Principal Component Analysis, we will assume that the data set has been treated for outliers either
from deletion of rows, or imputation of some other value.

That said, we will look at some methods to identify them.




'''

columns = ccfraud_df.iloc[: ,0:29]

for col in columns:
    plt.figure()
    plt.title('Box plot for column: ' + col)
    plt.boxplot(ccfraud_df[col])
    plt.show()



plt.boxplot(ccfraud_df['amount'])
plt.title("Boxplot for Amount")
plt.show()


plt.boxplot(ccfraud_df['time'])
plt.title("Boxplot for time")
plt.show()


# QQ plots for looking at distribution.

sm.qqplot(ccfraud_df['amount'], line ='45')
py.title('amount')
py.show()


sm.qqplot(ccfraud_df['time'], line ='45')
py.title('time')
py.show()

# Histogram for time
plt.hist(ccfraud_df['time'], bins = 20)



'''
###############################################################################
Univariate Outliers. IQR Method Detection
###############################################################################
Here we will use the IQR method to detect outliers.


A for loop goes through each column, calculates the IQR, then returns the values that are 1.5 times 
greater than the 0.75 quartile, or 1.5 times lesser than the 0.25 quartile. 
'''
# Detection of Univariate outliers using IQR method
# Create deep copy of data frame.
sss_train_IQR = sss_train.copy(deep = True)
sss_test_IQR = sss_test.copy(deep = True)


# Define function that calculates the IQR as well as upper and lower range.
#Upper for values greater than 1.5 times the 3rd quartile
def iqr_outlier_upper(df):
    outliers_upper = df[df>df.quantile(0.75) + (1.5 * (df.quantile(0.75) - df.quantile(0.25)))]
    return outliers_upper

#Lower for values lesser than 1.5 times the 1st quartile
def iqr_outlier_lower(df):
    outliers_lower = df[df<df.quantile(0.25) - (1.5 * (df.quantile(0.75) - df.quantile(0.25)))]
    return outliers_lower




for column in sss_train_IQR.drop(['class'], axis = 1):
    print("sss_train_IQR Upper-Outliers count for", column, ":",
          len(iqr_outlier_upper(sss_train_IQR[column])))

for column in sss_train_IQR.drop(['class'], axis = 1):
    print("sss_train_IQR Lower-Outliers count for", column, ":",
          len(iqr_outlier_lower(sss_train_IQR[column])))
    
 
    
    
for column in sss_test_IQR.drop(['class'], axis = 1):
      print("sss_test_IQR Upper-Outliers count for", column, ":",
            len(iqr_outlier_upper(sss_test_IQR[column])))

for column in sss_test_IQR.drop(['class'], axis = 1):
      print("sss_test_IQR Lower-Outliers count for", column, ":",
            len(iqr_outlier_lower(sss_test_IQR[column])))


print('Training set count of upper outliers is :', len(iqr_outlier_upper(sss_train_IQR['amount'])))
print('Training set count of lower outliers is :', len(iqr_outlier_lower(sss_train_IQR['amount'])))


print('Test set count of upper outliers is :', len(iqr_outlier_upper(sss_test_IQR['amount'])))
print('Test set count of lower outliers is :', len(iqr_outlier_lower(sss_test_IQR['amount'])))



###############################################################################
# Old code to remove outliers
###############################################################################
# Function to remove outliers using IQR method.
#def remove_outlier_IQR(df):
#    Q1 = df.quantile(0.25)
#    Q3 = df.quantile(0.75)
#    IQR = Q3 - Q1
#    df_final = df[~((df<(Q1 - 1.5*IQR))| (df > (Q3 + 1.5*IQR)))]
#    return df_final

#cc_outlier_removed = remove_outlier_IQR(cc2.amount)
#cc_outlier_removed = pd.DataFrame(cc_outlier_removed)
#ind_diff = cc2.index.difference(cc_outlier_removed.index)

#for i in range(0, len(ind_diff),1):
#    cc2 = cc2.drop([ind_diff[i]])
#    cc3 = cc2




'''
===============================================================================
IQR Cap method
===============================================================================
Similar to the IQR method of identifying outliers. Here we use the cap method to replace outlier
values with the maximum or minimum cut off.

'''
 

def IQR_cap_upper(column): #Use on selected column
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr_test = 1.5 * (q3 - q1)
    outlier_up_cutoff = column.quantile(0.75) + iqr_test
    return outlier_up_cutoff



def IQR_cap_lower(column): #Use on selected column
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr_test = 1.5 * (q3 - q1)
    outlier_low_cutoff = column.quantile(0.25) - iqr_test
    return outlier_low_cutoff




def df_upper_to_nan(df, column, selected_value):
    df[column].mask(df[column] > selected_value, np.nan, inplace = True)
    return df


def df_lower_to_nan(df, column, selected_value):
    df[column].mask(df[column] < selected_value, np.nan, inplace = True)
    return df

# Deep frame
sss_train_CAP = sss_train.copy(deep = True)
sss_test_CAP = sss_test.copy(deep = True)


# Cap for training set - upper and lower
train_amount_upper_cutoff = round(IQR_cap_upper(sss_train_CAP['amount']),2)
train_amount_lower_cutoff = round(IQR_cap_lower(sss_train_CAP['amount']),2)



df_upper_to_nan(sss_train_CAP, 'amount', train_amount_upper_cutoff)
df_lower_to_nan(sss_train_CAP, 'amount', train_amount_lower_cutoff)


na_check(sss_train_CAP)
# Upper Training
sss_train_CAP['amount'] = sss_train_CAP['amount'].fillna(train_amount_upper_cutoff)
sss_train_CAP['amount'].max() == train_amount_upper_cutoff


# Lower Training
sss_train_CAP['amount'] = sss_train_CAP['amount'].fillna(train_amount_lower_cutoff)
sss_train_CAP['amount'].min() == train_amount_lower_cutoff


# Cap for test set - upper and lower
test_amount_upper_cutoff = round(IQR_cap_upper(sss_test_CAP['amount']),2)
test_amount_lower_cutoff = round(IQR_cap_lower(sss_test_CAP['amount']),2)

df_upper_to_nan(sss_test_CAP, 'amount', test_amount_upper_cutoff)
df_lower_to_nan(sss_test_CAP, 'amount', test_amount_lower_cutoff)


na_check(sss_train_CAP)
# Upper Test
sss_test_CAP['amount'] = sss_test_CAP['amount'].fillna(test_amount_upper_cutoff)
sss_test_CAP['amount'].max() == test_amount_upper_cutoff


# Lower Test
sss_test_CAP['amount'] = sss_test_CAP['amount'].fillna(test_amount_lower_cutoff)
sss_test_CAP['amount'].min() == train_amount_lower_cutoff # return false value, as no lower cap found

'''
==============================================================================
Imputation method - Using Imputer from sklearn.
==============================================================================
In the below code, we replace outlier values in the "amount" column with NAN values,
We then use Iterative imputer from sci-kit learn.

A more sophisticated approach is to use the IterativeImputer class, which models each feature with 
missing values as a function of other features, and uses that estimate for imputation. 
It does so in an iterated round-robin fashion: at each step, a feature column is designated as 
output y and the other feature columns are treated as 
inputs X. A regressor is fit on (X, y) for known y. 
Then, the regressor is used to predict the missing values of y. 
This is done for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. 
The results of the final imputation round are returned.

'''

X_train_IMP = X_train.copy(deep = True)
X_test_IMP = X_test.copy(deep = True)



# TRAINING SET - Cap for training set - upper and lower
train_amount_upper_cutoff_IMP = round(IQR_cap_upper(X_train_IMP['amount']),2)
train_amount_lower_cutoff_IMP = round(IQR_cap_lower(X_train_IMP['amount']),2)

# TRAINING SET - turn outliers to NAN values
df_upper_to_nan(X_train_IMP, 'amount', train_amount_upper_cutoff_IMP)
df_lower_to_nan(X_train_IMP, 'amount', train_amount_lower_cutoff_IMP)




# TEST SET - Cap for test set - upper and lower
test_amount_upper_cutoff_IMP = round(IQR_cap_upper(X_test_IMP['amount']),2)
test_amount_lower_cutoff_IMP = round(IQR_cap_lower(X_test_IMP['amount']),2)

# TEST SET - turn outliers to NAN values
df_upper_to_nan(X_test_IMP, 'amount', test_amount_upper_cutoff_IMP)
df_lower_to_nan(X_test_IMP, 'amount', test_amount_lower_cutoff_IMP)



###############################################################################
#########                  Imputer                               ##############

# set random state
imputer = IterativeImputer(random_state = 420, n_nearest_features = None,
                           imputation_order = 'ascending')

###############################################################################


# For Training Set
imputer.fit(X_train_IMP)
X_train_IMP = imputer.transform(X_train_IMP)
X_train_IMP = pd.DataFrame(X_train_IMP, columns = X_train.columns)
X_train_IMP['amount'] = round(X_train_IMP['amount'],2)


na_check(X_train_IMP)

# For Testing set:
imputer.fit(X_test_IMP)
X_test_IMP = imputer.transform(X_test_IMP)
X_test_IMP = pd.DataFrame(X_test_IMP, columns = X_test.columns)
X_test_IMP['amount'] = round(X_test_IMP['amount'],2)


na_check(X_test_IMP)











### Delete
#Check nan counts
#na_check(sss_train_CAP)

#sss_train_CAP['amount'] = sss_train_CAP['amount'].fillna(train_amount_upper_cutoff)
#sss_train_CAP['amount'].max() == train_amount_upper_cutoff






'''
###############################################################################
Multivariate Outlier Detection:- One Class SVM
###############################################################################
https://towardsdatascience.com/support-vector-machine-svm-for-anomaly-detection-73a8d676c331

OneClassSVM is a unsupervised machine learning algorithm that learns what the distribution of the features should 
be from the data itself, and therefore is applicable in a large variety of situations when you want 
to be able to catch all the outliers but also the unusual data examples.

In order to have OneClassSVM work as an outlier detector, you need to work on its core parameters;
it requires you to define the kernel, degree, gamma, and nu: 

Kernel and degree: These are interconnected. Usually, the values that we suggest based on our 
experience are the default ones; the type of kernel should be rbf and its degree should be 3. 
Such parameters will inform OneClassSVM to create a series of classification bubbles that span 
through three dimensions, allowing you to model even the most complex multidimensional distribution forms. 

Gamma: This is a parameter that's connected to the RBF kernel. We suggest that you keep it as 
low as possible. A good rule of thumb should be to assign it a minimum value that lies between the 
inverse of the number of cases and the variables. 
Higher values of gamma tend to lead the algorithm to follow the data, but more so define the shape 
of the classification bubbles. 

Nu: This parameter determines whether we have to fit the exact distribution or if we try to obtain 
a certain degree of generalization by not adapting too much to the present data examples 
(a necessary choice if outliers are present). It can be easily determined with the help of the following formula: 
    nu_estimate = 0.95 * outliers_fraction + 0.05 If the value of the outliers' 
    fraction is very small, nu will be small and the SVM algorithm will try to fit the contour of 
    the data points. On the other hand, if the fraction is high, so will the parameter be, forcing 
    a smoother boundary of the inliers' distributions.



After running the algorithm we print out the outliers in our training and test set.
'''
# Create Deep copies
sss_train_SVM = sss_train.copy(deep = True)
sss_test_SVM = sss_test.copy(deep = True)




# SVM DETECTION ON TRAINING SET

# SCALE AMOUNT AND TIME COLUMNS USING ROBUST SCALER
#std_scaler = StandardScaler() not needed.
rob_scaler = RobustScaler()

# Creates new dataframe column with scaled values for amount and time
sss_train_SVM['scaled_amount'] = rob_scaler.fit_transform(sss_train_SVM['amount'].values.reshape(-1,1))
sss_train_SVM['scaled_time'] = rob_scaler.fit_transform(sss_train_SVM['time'].values.reshape(-1,1))

sss_train_SVM.drop(['time','amount'], axis=1, inplace=True)



# Drops scaled amount and scaled columns and inserts them in first 2 columns.
scaled_amount = sss_train_SVM['scaled_amount']
scaled_time = sss_train_SVM['scaled_time']

sss_train_SVM.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
sss_train_SVM.insert(0, 'scaled_amount', scaled_amount)
sss_train_SVM.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

sss_train_SVM.head()


# TRAINING SET - Model Specification

# OneClassSVM fitting and estimates
outliers_fraction = 0.02 # 
nu_estimate = 0.95 * outliers_fraction + 0.05
machine_learning = OneClassSVM(kernel="rbf",
                                   gamma=1.0/len(sss_train_SVM),
                                   degree=3, nu=nu_estimate)
machine_learning.fit(sss_train_SVM)
detection = machine_learning.predict(sss_train_SVM)
outliers_train = np.where(detection==-1)
regular_train = np.where(detection==1)


outlier_values_train = sss_train_SVM.iloc[outliers_train]
outlier_values_train




###################################################################################################
#OLD
#model = OneClassSVM(kernel = 'rbf', gamma = 0.001, nu = 0.03).fit(sss_train_SVM)


# Prediction
#y_pred_train = model.predict(sss_train_SVM)
#y_pred_train



# filter outlier index
#outlier_index = y_pred_train == -1
# filter outlier values
#outlier_values_train = sss_train_SVM.iloc[outlier_index]
#outlier_values_train
####################################################################################################



# SVM DETECTION ON TESTING SET
sss_test_SVM = sss_test.drop(['class'], axis=1)


# SCALE AMOUNT AND TIME COLUMNS USING ROBUST SCALER
std_scaler = StandardScaler()
rob_scaler = RobustScaler()

# Creates new dataframe column with scaled values for amount and time
sss_test_SVM['scaled_amount'] = rob_scaler.fit_transform(sss_test_SVM['amount'].values.reshape(-1,1))
sss_test_SVM['scaled_time'] = rob_scaler.fit_transform(sss_test_SVM['time'].values.reshape(-1,1))

sss_test_SVM.drop(['time','amount'], axis=1, inplace=True)



# Drops scaled amount and scaled columns and inserts them in first 2 columns.
scaled_amount = sss_test_SVM['scaled_amount']
scaled_time = sss_test_SVM['scaled_time']

sss_test_SVM.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
sss_test_SVM.insert(0, 'scaled_amount', scaled_amount)
sss_test_SVM.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

sss_test_SVM.head()


# TRAINING SET - Model Specification

# OneClassSVM fitting and estimates
outliers_fraction = 0.02 # 
nu_estimate = 0.95 * outliers_fraction + 0.05
machine_learning = OneClassSVM(kernel="rbf",
                                   gamma=1.0/len(sss_test_SVM),
                                   degree=3, nu=nu_estimate)
machine_learning.fit(sss_train_SVM)
detection = machine_learning.predict(sss_test_SVM)
outliers_test = np.where(detection==-1)
outliers_test = np.where(detection==1)


outlier_values_train = sss_test_SVM.iloc[outliers_test]
outlier_values_train











'''
###############################################################################
Data Modelling
###############################################################################
In building models to predicts which transactions are fraud and which are legitimate, we will use 
the following machine learning algorithms:
- KNN K Nearest Neighbour
- Decision Trees
- XGBoost


We use the following metrics to evaluate the performance of our model:
    
- Average Precision
Like the Area under the Precision-Recall curve (AUC-PR) metric, Average Precision is a way 
to summarize the PR curve into a single value. To define the term, the Average Precision 
metric (or just AP) is the weighted mean of Precision scores achieved at each PR curve threshold, 
with the increase in Recall from the previous threshold used as the weight. 


- Training Time

- Prediction Time



- f1 Score
Is an alternative machine learning evaluation metric that assesses the predictive skill of a model 
by elaborating on its class-wise performance rather than an overall performance as done by accuracy. 
F1 score combines two competing metrics- precision and recall scores of a model, leading to its 
widespread use in recent literature.

The F1 score combines precision and recall using their harmonic mean, and maximizing the F1 score 
implies simultaneously maximizing both precision and recall. It is recommended for use in
datasets with imbalanced data such as fraud.


- f beta score
An extension of f1 score. The beta parameter represents the ratio of recall importance 
to precision importance. beta > 1 gives more weight to recall, while beta < 1 favors precision.
#https://medium.com/@douglaspsteen/beyond-the-f-1-score-a-look-at-the-f-beta-score-3743ac2ef6e3#:~:text=The%20F%2Dbeta%20score%20calculation,to%20the%20F%2D1%20Score.

When we care more about minimizing false positives than minimizing false negatives,
we would want to select a beta value of < 1 for the F-beta score. In other words, 
precision would be given more weight than recall in this scenario. 


On the other hand, when the priority is to minimize false negatives, we would 
want to select a beta value of >1 for the F-beta score. Recall would be considered more 
important than precision in this scenario. 
###############################################################################


###############################################################################
KNN
###############################################################################
K-Nearest Neighbors, or simply k-NN, belongs to the class of instance-based learning, 
also known as lazy classifiers. It's one of the simplest classification methods because 
the classification is done by just looking at the K-closest examples in the training set 
(in terms of Euclidean distance or some other kind of distance) in the case that we want 
to classify. 
Then, given the K-similar examples, the most popular target (majority voting) is chosen as the 
classification label. 

Two parameters are mandatory for this algorithm: the neighborhood cardinality (K), and 
the measure to evaluate the similarity (although the Euclidean distance, or L2, is the 
most used and is the default parameter for most implementations).


We use GridSearch to determine best paramaters of p and K, we set cross validation to 5.

This returns the following parameters:
p = 1 - this is equivalent to Manhattan Distance
k = 11


Our results from below:
    
Best p: 1
Best n_neighbors: 11
average_precision 0.6761672314497005
training time 0.08161332130432128
prediction time 0.0003154945026969756
KNN f_1score is:  0.813953488372093
KNN fbeta_0_8 is:  0.8396723229959041
KNN fbeta_1_5 is:  0.7724957555178268
KNN fbeta_2_0 is:  0.7510729613733906
'''



from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score


cf_column_transformer = ColumnTransformer(transformers=[('Robust', RobustScaler(), [0,29])], 
                                          remainder='passthrough')


pipeline = Pipeline(steps=[('cf_column_transformer', cf_column_transformer),
                           ('KNeighborsClassifier', KNeighborsClassifier())])


param_grid = {'KNeighborsClassifier__n_neighbors': [5, 7, 9, 11, 13],
              'KNeighborsClassifier__p': [1,2]}


knn = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring = 'average_precision')


knn.fit(X_train, y_train)

print('Best p:', knn.best_estimator_.get_params()['KNeighborsClassifier__p']) 

print('Best n_neighbors:', knn.best_estimator_.get_params()['KNeighborsClassifier__n_neighbors']) 




results=pd.DataFrame() 
y_predicted = knn.predict(X_test)



#Average Precision
results.loc['knn','average_precision']=average_precision_score(y_test, y_predicted) 
print('average_precision', results.loc['knn','average_precision']) 

#Training Time in Seconds
results.loc['knn','training_time']=knn.cv_results_['mean_fit_time'].mean() 
print('training time', results.loc['knn','training_time']) 


# Prediction Time in Seconds
results.loc['knn','prediction_time']=knn.cv_results_['mean_score_time'].mean()/len(y_test) 
print('prediction time', results.loc['knn','prediction_time']) 


'''RESULTS'''
#Best p: 1
#Best n_neighbors: 13
#training time 0.09892102718353273
#prediction time 0.00024870174595204757

###############################################################################
#                     ''' f beta measure'''
###############################################################################

from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score



results.loc['knn', 'f1_score'] = f1_score(y_test,y_predicted)
print('KNN f_1score is: ',results.loc['knn','f1_score'])


#If reducing false negatives,we would want to select a beta value of < 1
results.loc['knn','fbeta_0_8']= fbeta_score(y_test,y_predicted, beta = 0.8)
print(results.loc['knn','fbeta_0_8'])
print('KNN fbeta_0_8 is: ',results.loc['knn','fbeta_0_8'])


#If minimize false negatives, we would want to select a beta value of > 1
results.loc['knn','fbeta_1_5']= fbeta_score(y_test,y_predicted, beta = 1.5)
print(results.loc['knn','fbeta_1_5'])
print('KNN fbeta_1_5 is: ',results.loc['knn','fbeta_1_5'])

#If minimize false negatives, we would want to select a beta value of > 1
results.loc['knn','fbeta_2_0']= fbeta_score(y_test,y_predicted, beta = 2.0)
print(results.loc['knn','fbeta_2_0'])
print('KNN fbeta_2_0 is: ',results.loc['knn','fbeta_2_0'])


'''
###############################################################################
Decision Trees
###############################################################################
Decision Trees (DTs) are a non-parametric supervised learning method used for classification 
and regression. 

The goal is to create a model that predicts the value of a target variable by learning simple 
decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

While being effective for variables of the categorical and continuous input and output, a decision 
tree breaks the samples or population into two or more equivalent groups (or sub-populations) which 
depends on the most significant splitter (or differentiator) in input variables (Ray Sunil 2015).
A decision tree is presented as a structure of a flowchart. It demonstrates one approach of how we 
make decisions whose logic is similar to "if-this-then-that". 

In a decision tree,
•	each non-leaf/internal node represents a "test" on an attribute or feature, questions asked 
     are binary questions which can have answers like True/False, Yes/No,
•	each branch represents the outcome of the test, and
•	each leaf node represents a class label (decision taken after computing all attributes).



Again, we use Gridsearch to determine the best parameters, setting cross validation to 5.
- Best max_depth: 5
Max Depth is the maximum depth of the tree. Used to control overfitting as higher depth will 
allow model to learn relations very specific to a particular sample.


- Best min_samples_leaf: 20
o	Defines the minimum samples (or observations) required in a terminal node or leaf.
o	Used to control overfitting similar to min_samples_split.
o	Generally lower values should be chosen for imbalanced class problems, because the regions in 
    which the minority class will be in majority will be very small.





Best Criterion: Gini Index
Gini index provides a measurement for the scale or possibility of a specific variable being 
mistakenly organised into groups when it is randomly selected. 
When two items are randomly picked from a sample, the probability of them falling into the same 
class is 1 if the sample is 'pure'.
Thus, Gini index can be used as a splitting measure to create binary splits for a decision tree. 
The closer the Gini index is towards 1, the higher the homogeneity is.

###################################################################################################
Decision Tree Results
####################################################################################################
dt average_precision is:  0.6154664866925882
dt training_time is:  10.562634914398194
dt prediction_time is:  2.7117969104025764e-07

dt f1_score is:  0.7796610169491525
dt fbeta_0_8 is:  0.7984758679085521
dt fbeta_1_5 is:  0.7487479131886478
dt fbeta_2_0 is:  0.7324840764331211




'''


from sklearn.tree import DecisionTreeClassifier 



cf_column_transformer = ColumnTransformer(transformers=[('Robust', RobustScaler(), [0,29])], 
                                          remainder='passthrough')







pipeline = Pipeline(steps=[('cf_column_transformer', cf_column_transformer),
                           ('dt', DecisionTreeClassifier())])

param_grid = { 

    'dt__min_samples_split': [2 , 3, 5], 

    'dt__min_samples_leaf': [5, 10, 20, 50, 100], 

    'dt__max_depth': [2, 3, 5, 10, 20] 

} 

dt = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring = 'average_precision')



dt.fit(X_train, y_train) 

print('Best max_depth:', dt.best_estimator_.get_params()['dt__max_depth']) 

print('Best min_samples_leaf:', dt.best_estimator_.get_params()['dt__min_samples_leaf']) 

print('Best criterion:', dt.best_estimator_.get_params()['dt__criterion']) 


y_predicted = dt.predict(X_test)




#Average Precision
results.loc['dt','average_precision']=average_precision_score(y_test, y_predicted) 
print('average_precision', results.loc['dt','average_precision']) 
print('dt average_precision is: ',results.loc['dt','average_precision'])


# Training Time (in seconds)
results.loc['dt','training_time']=dt.cv_results_['mean_fit_time'].mean() 
print(results.loc['dt','training_time']) 
print('dt training_time is: ',results.loc['dt','training_time'])


# Prediction time (in seconds)
results.loc['dt','prediction_time']=dt.cv_results_['mean_score_time'].mean()/len(y_test) 
print(results.loc['dt','prediction_time']) 
print('dt prediction_time is: ',results.loc['dt','prediction_time'])



#Best max_depth: 5
#Best min_samples_leaf: 20
#Best criterion: gini
#6.731390288670858
#3.17109294826308e-07





results.loc['dt', 'f1_score'] = f1_score(y_test,y_predicted)
print(results.loc['dt','f1_score'])
print('dt f1_score is: ',results.loc['dt','f1_score'])

#If reducing false negatives,we would want to select a beta value of < 1

results.loc['dt','fbeta_0_8']= fbeta_score(y_test,y_predicted, beta = 0.8)
print(results.loc['dt','fbeta_0_8'])
print('dt fbeta_0_8 is: ',results.loc['dt','fbeta_0_8'])


#If minimize false negatives, we would want to select a beta value of > 1
results.loc['dt','fbeta_1_5']= fbeta_score(y_test,y_predicted, beta = 1.5)
print(results.loc['dt','fbeta_1_5'])
print('dt fbeta_1_5 is: ',results.loc['dt','fbeta_1_5'])

#If minimize false negatives, we would want to select a beta value of > 1
results.loc['dt','fbeta_2_0']= fbeta_score(y_test,y_predicted, beta = 2.0)
print(results.loc['dt','fbeta_2_0'])
print('dt fbeta_2_0 is: ',results.loc['dt','fbeta_2_0'])
'''
###############################################################################
XGBoost
###############################################################################
The XGBoost library implements the gradient boosting decision tree algorithm.

This algorithm goes by lots of different names such as gradient boosting, multiple 
additive regression trees, stochastic gradient boosting or gradient boosting machines.

Boosting is an ensemble technique where new models are added to correct the errors made by existing 
models. Models are added sequentially until no further improvements can be made. A popular example 
is the AdaBoost algorithm that weights data points that are hard to predict.

Gradient boosting is an approach where new models are created that predict the residuals or 
errors of prior models and then added together to make the final prediction. It is called gradient 
boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

This approach supports both regression and classification predictive modeling problems.



Again Using Gridsearch and setting cross validation to 5 gives us the following parameters:
Best max_depth: 3
Best min_child_weight: 0.8
Best gamma: 0.02
Best subsample: 0.9
Best col_sample by tree: 0.5
Best learning rate: 0.1
Best n_estimators: 500


-Results of model
xgb average_precision is:  0.7159329814577424
xgb training_time is:  10.562634914398194
xgb prediction_time is:  2.7117969104025764e-07
xgb f1_score is:  0.839080459770115
xgb fbeta_0_8 is:  0.863033448673587
xgb fbeta_1_5 is:  0.8001686340640809
xgb fbeta_2_0 is:  0.7799145299145299




'''
# https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://medium.com/@rithpansanga/the-main-parameters-in-xgboost-and-their-effects-on-model-performance-4f9833cac7c

import xgboost as xgb
from sklearn.metrics import fbeta_score


param_grid = {
    'xgb__max_depth': [3, 6],
    'xgb__min_child_weight': [0.8, 1],
    'xgb__gamma': [0.01, 0.02, 0.03],
    'xgb__subsample': [0.90],
    'xgb__colsample_bytree': [0.5, 0.8],
    'xgb__learning_rate': [0.001, 0.01, 0.1],
    'xgb__n_estimators': [200,500]
    
    }


pipeline = Pipeline(steps=[('cf_column_transformer', cf_column_transformer),
                           ('xgb', xgb.XGBClassifier(objective = 'multi:softmax', num_class = 2))])


xgb = GridSearchCV(pipeline, param_grid = param_grid, cv=5,
                   scoring='average_precision')

#xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='aucpr', early_stopping_rounds = 25, verbose = False)


xgb.fit(X_train, y_train)
print('Best max_depth:', xgb.best_estimator_.get_params()['xgb__max_depth'])
print('Best min_child_weight:', xgb.best_estimator_.get_params()['xgb__min_child_weight'])
print('Best gamma:', xgb.best_estimator_.get_params()['xgb__gamma'])
print('Best subsample:', xgb.best_estimator_.get_params()['xgb__subsample'])
print('Best col_sample by tree:', xgb.best_estimator_.get_params()['xgb__colsample_bytree'])
print('Best learning rate:', xgb.best_estimator_.get_params()['xgb__learning_rate'])
print('Best n_estimators:', xgb.best_estimator_.get_params()['xgb__n_estimators'])


y_predicted = xgb.predict(X_test)


#Best max_depth: 6
#Best min_child_weight: 1
#Best gamma: 0.03
#Best subsample: 0.9
#Best learning rate: 0.1
#Best n_estimators: 500



#Average Precision
results.loc['xgb', 'average_precision'] = average_precision_score(y_test, y_predicted)
print(results.loc['xgb', 'average_precision'])
print('xgb average_precision is: ',results.loc['xgb','average_precision'])


# Training Time (in seconds)
results.loc['xgb','training_time']=dt.cv_results_['mean_fit_time'].mean() 
print(results.loc['xgb','training_time']) 
print('xgb training_time is: ',results.loc['xgb','training_time'])


# Prediction time (in seconds)
results.loc['xgb','prediction_time']=dt.cv_results_['mean_score_time'].mean()/len(y_test) 
print(results.loc['xgb','prediction_time']) 
print('xgb prediction_time is: ',results.loc['xgb','prediction_time'])

#Best max_depth: 6
#Best min_child_weight: 1
#Best gamma: 0.02
#Best subsample: 0.9
#Best learning rate: 0.1
#Best n_estimators: 200


results.loc['xgb', 'f1_score'] = f1_score(y_test,y_predicted)
print(results.loc['xgb','f1_score'])
print('xgb f1_score is: ',results.loc['xgb','f1_score'])

#If reducing false negatives,we would want to select a beta value of < 1

results.loc['xgb','fbeta_0_8']= fbeta_score(y_test,y_predicted, beta = 0.8)
print(results.loc['xgb','fbeta_0_8'])
print('xgb fbeta_0_8 is: ',results.loc['xgb','fbeta_0_8'])


#If minimize false negatives, we would want to select a beta value of > 1
results.loc['xgb','fbeta_1_5']= fbeta_score(y_test,y_predicted, beta = 1.5)
print(results.loc['xgb','fbeta_1_5'])
print('xgb fbeta_1_5 is: ',results.loc['xgb','fbeta_1_5'])


#If minimize false negatives, we would want to select a beta value of > 1
results.loc['xgb','fbeta_2_0']= fbeta_score(y_test,y_predicted, beta = 2.0)
print(results.loc['xgb','fbeta_2_0'])
print('xgb fbeta_2_0 is: ',results.loc['xgb','fbeta_2_0'])



"""
###############################################################################



"""


#https://medium.com/@douglaspsteen/beyond-the-f-1-score-a-look-at-the-f-beta-score-3743ac2ef6e3#:~:text=The%20F%2Dbeta%20score%20calculation,to%20the%20F%2D1%20Score.




#If minimize false negatives, we would want to select a beta value of > 1
