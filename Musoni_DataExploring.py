# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 13:06:57 2016

@author: AGGREY
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Load dataset
class LoadDataset(object):
    """Loads dataset from CSV file"""
    
    def __init__(self):
        pass
    
    def load_file(self, file_name, required_columns=None):
        if(required_columns is None):
            dataset = pd.read_csv(file_name, low_memory=False)
        else:
            dataset = pd.read_csv(file_name, usecols=required_columns, low_memory=False)       
        return dataset


def load_dataset(file_name):
    return LoadDataset().load_file(file_name,required_columns=None)

file_name = 'loans_clients_officers_location_combined.csv'
df = load_dataset(file_name)

df['gender'] = df['gender'].replace(54,'female')
df['gender'] = df['gender'].replace(55, 'male')

def branches_vs_savings():
    df_table = pd.pivot_table(data = df, index=['office_name'], values=['approved_principal'], aggfunc='sum')
    df_table = df_table.sort_values(by='approved_principal') # sort the table in ascending order
    print(df_table)
    x_labels = df_table.index.tolist()
    y_axis = (df_table['approved_principal']/1000000).tolist()
    x_axis = np.arange(len(y_axis))
    sns.set(style="white")
    sns.barplot(x_axis, y_axis) #Plot
    sns.despine();
    plt.xticks(x_axis,x_labels, rotation='90')#Add label
    plt.xlabel('Branch Names')
    plt.ylabel('Loan Amount(Millions)')
    plt.title('Amount Of Loans(In Millions) Vs Branch Names')
    plt.show()

#branches_vs_savings()


def gender_countOfLoans():
    df_table = pd.pivot_table(data = df, index=['gender'], values=['approved_principal'], aggfunc='count')
    df_table = df_table.sort_values(by='approved_principal') # sort the table in ascending order
    x_labels = df_table.index.tolist()
    y_axis = (df_table['approved_principal']).tolist()
    x_axis = np.arange(len(y_axis))
    plt.bar(x_axis, y_axis, color=['orange','green'], alpha=0.9, width=0.5, align='center')
    plt.xlabel('Gender Of Receipients')
    plt.ylabel('Number Of Loans')
    plt.title('Number Of Loans Vs Gender Of Loan Receipients')
    plt.xticks(x_axis,x_labels)

#gender_countOfLoans()

def office_numberOfLoans():
    df_table = pd.pivot_table(data = df, index=['office_name'], values=['approved_principal'], aggfunc='count')
    df_table = df_table.sort_values(by='approved_principal')
    print(df_table)
    x_labels = df_table.index.tolist()
    y_axis = (df_table['approved_principal']).tolist()
    x_axis = np.arange(len(y_axis))
    sns.set(style="white")
    sns.barplot(x_axis, y_axis) #Plot
    sns.despine()
    plt.xlabel('Office Name', size=12)
    plt.ylabel('Number Of Loans Disbursed', size=12)
    plt.title('Number Of Loans Disbursed Vs Office Names')
    plt.xticks(x_axis,x_labels, rotation='90')

#office_numberOfLoans()

# NEXT PLOT THE NUMBER OF OFFICERS IN EACH OFFICE. THIS SHOULD BUILD UP THE CO-RELATION
def loans_officers():
    #Load Loan Officers' details
    file3_name = 'musoni/loan_officers.csv'
    required_columns=[0,2,7]
    loansOfficers = LoadDataset().load_file(file3_name, required_columns)
    loansOfficers.rename(columns = {'id' : 'loan_officer_id','external_id' : 'loan_officer_name'}, inplace=True)
    return loansOfficers

# Load Office Locations
def office_locations():
    file4_name = 'musoni/m_office.csv'
    required_columns = [0,4]
    officeLocations = LoadDataset().load_file(file4_name, required_columns)
    officeLocations.rename(columns={'id':'office_id', 'name':'office_name'}, inplace=True)
    return officeLocations

#Join Loans Officers and the Offices they are attached to
def offices_officeLocation(loan_officers, office_locations):
    merged = loan_officers.merge(office_locations, on='office_id')
    return merged
    
def office_numberOfLoanOfficers():
    loan_officers = loans_officers()
    offices= office_locations()
    staff_and_offices_df = offices_officeLocation(loan_officers, offices)
    df_table = pd.pivot_table(data = staff_and_offices_df, index=['office_name'], values=['loan_officer_id'], aggfunc='count')
    df_table = df_table.sort_values(by='loan_officer_id')
    print(df_table)
    x_labels = df_table.index.tolist()
    y_axis = (df_table['loan_officer_id']).tolist()
    x_axis = np.arange(len(y_axis))
    sns.set(style="white")
    sns.barplot(x_axis, y_axis) #Plot
    sns.despine()
    plt.xlabel('Office Name', size=12)
    plt.ylabel('Number Of Loan Officers', size=12)
    plt.title('Number Of Loan Officers Vs Office Name')
    plt.xticks(x_axis,x_labels, rotation='90')

#office_numberOfLoanOfficers()

df_table = pd.pivot_table(data = df, index=['office_name', 'loan_officer_name'], values=['approved_principal'])
#office_numberOfLoans() 
print(df_table)  
