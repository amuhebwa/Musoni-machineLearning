# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:26 2016

@author: AGGREY
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Plt distribution of principal
def plot_principal(trainData) :
    plt.hist(trainData['Average Principal'], alpha=1.0, bins=20)
    plt.title('Distribution of Average Principal')
    plt.ylabel('Number of Loans')
    plt.xlabel('Average Principle')
    fig = plt.gcf()
    plt.show()
    fig.savefig('graphs/average_principal.png')

#Plot distribution of days
def plot_averageDaysOverdue(df_train):
    sns.kdeplot(df_train['Average Over Due Days'], shade=True, color='g')
    plt.title('Estimate of average days overdue as of December')
    plt.xlabel('Average of days overdue')
    plt.ylabel('Probability Distribution')
    fig = plt.gcf()
    plt.show()
    fig.savefig('graphs/average_overdue_days.png')


# Plot average over due as of the previous one month
def plot_averageAsOfLastMonth(df_train):
    sns.kdeplot(df_train['Past One Month'], shade=True, color='r')
    plt.title('Estimate of average days overdue as of past one month')
    plt.xlabel('Average Of days ovedue')
    plt.ylabel('Probability Distribution')
    fig = plt.gcf()
    plt.show()
    fig.savefig('graphs/past_one_month.png')


# Plot average over due as of the previous three months
def plot_averageAsOfPastThreeMonths(df_train):
    sns.kdeplot(df_train['Past Three Months'], shade=True, color='purple')
    plt.title('Estimate of average days overdue as of past three months')
    plt.xlabel('Average of days overdue')
    plt.ylabel('Probabiity Distribution')
    fig = plt.gcf()
    plt.show()
    fig.savefig('graphs/past_three_months.png')


#Load training Data
file_train = 'groups_final_dataset.csv'
df_train = pd.read_csv(file_train, low_memory=False)   
trainData = df_train.drop(['Average Over Due Days'], axis=1)
trainData = trainData.sample(frac=1).reset_index(drop=True) #Randomize
x_train = trainData[['Number Of Loans','Average Principal', 'Past One Month','Past Three Months']]
y_train = trainData[['Low/High Risk']]

#plot_principal(trainData)
#plot_averageDaysOverdue(df_train)
#plot_averageAsOfLastMonth(df_train)
#plot_averageAsOfPastThreeMonths(df_train)

#LOAD TESTING DATA
file_test = 'groups_final_dataset_test.csv'
df_test = pd.read_csv(file_test, low_memory=False)   
#testData = df_test.drop(['Average Over Due Days'], axis=1)
x_test = df_test[['Number Of Loans','Average Principal', 'Past One Month','Past Three Months']]


forest= RandomForestClassifier(n_estimators=100, criterion='gini', oob_score=True, max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
forest.fit(x_train, y_train.values.ravel())
results= forest.predict(x_test)
print("\n\nPredicted outcome\n")
print(results)
print('%Error', forest.oob_prediction)



