# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 18:01:49 2016

@author: AGGREY
"""

import numpy as np
import pandas as pd


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

# LOAD THE LOANS INFORMATION
def load_loans():
    file_name = 'musoni/m_loan.csv'
    required_columns = [0,3,16] 
    loans = LoadDataset().load_file(file_name,required_columns)
    loans.rename(columns={'id':'loan_id', 'closedon_date':'loanClosedDate'}, inplace=True)
    return loans;


# LOAD M_GROUP_CLIENTS
def load_groupClients():
    file1_name = 'musoni/m_group_client.csv'
    group_client = LoadDataset().load_file(file1_name,required_columns = None)
    return group_client

# LOAD GROUPS
def load_groups():
    file2_name = 'musoni/m_groups.csv'
    required_columns = [0,8]
    groups =  LoadDataset().load_file(file2_name,required_columns)
    groups.rename(columns={'id':'group_id', 'display_name':'group_name', 'closedon_date':'groupClosedDate'}, inplace=True)
    return groups

#MERGE GROUPS, CLIENTS AND LOANS IN ORDER TO GET THE GROUPS IN WHICH THE CLIENTS WHO TOOK LOANS BELONG
def merge_all():
    groups_clients = load_groupClients()
    groups = load_groups()
    groups_clients_merged = groups.merge(groups_clients, on='group_id')
    loans = load_loans();
    groups_clients_loans_merged = loans.merge(groups_clients_merged, on='client_id')
    #groups_clients_loans_merged.to_csv('groups_clients_loans_merged.csv', index=False)
    #print('Finished merging files')
    return groups_clients_loans_merged


def loan_evaluation():
    file_name = 'musoni/loan_balance_evolution.csv'
    required_columns = [0,1,35]
    df =  LoadDataset().load_file(file_name,required_columns)
    df = df.fillna(0)
    return df;
def generateFinalDataset():
    loans_eval = loan_evaluation()
    clientsGroups = merge_all()
    df = clientsGroups.merge(loans_eval, on = 'loan_id')
    df = df.sort_values(by='group_id')
    # pivot table with the generated values rounded to 2dp
    x = np.round(pd.pivot_table(data = df, index=['group_id'], values=['loan_id','days_overdue_2015','approved_principal'], aggfunc={'loan_id':'count','days_overdue_2015':np.mean,'approved_principal':np.mean}), 2)
    x = x.rename(columns={'days_overdue_2015':'Average Over Due Days', 'loan_id':'Number Of Loans', 'approved_principal':'Average Principal'})
    x = x[['Number Of Loans','Average Principal','Average Over Due Days']]  
    return x;

df = generateFinalDataset()

# Add a column  to indicate whether a group is risky or not
count = df['Average Over Due Days'].count()
sum =  df['Average Over Due Days'].sum()
threshold = np.round((sum/count), 2) # Average of all days rounded to 2dp
print'Threshold : ', threshold
# 1 - High Risk (very risky)
# 0 - Low Risk(none risky)
df['Low/High Risk'] = np.where(df['Average Over Due Days'] < threshold, 0, 1)
df.to_csv('groups_final_dataset.csv', index=False)
print("done")



