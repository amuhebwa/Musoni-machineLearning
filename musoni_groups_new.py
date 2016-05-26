# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:25:09 2016

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

def merge_all():
    groups_clients = load_groupClients()
    groups = load_groups()
    groups_clients_merged = groups.merge(groups_clients, on='group_id')
    loans = load_loans();
    groups_clients_loans_merged = loans.merge(groups_clients_merged, on='client_id')
    #groups_clients_loans_merged.to_csv('groups_clients_loans_merged.csv', index=False)
    #print('Finished merging files')
    return groups_clients_loans_merged

# p = merge_all()
def load_loanEvaluation():
    file_name = 'musoni/loan_balance_evolution.csv'
    required_columns = [0,1,35]
    df =  LoadDataset().load_file(file_name,required_columns)
    df = df.fillna(0)
    return df;
#q = load_loanEvaluation()
def loan_loanHistory():
    file_name1 = 'musoni/loan_overdue_history.csv'
    required_columns =[0,3,4,5]
    loan_history = LoadDataset().load_file(file_name1,required_columns)
    loan_history = loan_history.replace(np.nan,0, regex=True)
    return loan_history

def merge1():
    loans_eval = load_loanEvaluation()
    clients_groups = merge_all()
    loan_history = loan_loanHistory()
    # MERGE LOAN EVALUATION WITH LOANS HISTORY
    loan_History_Evaluation = loans_eval.merge(loan_history, on='loan_id')
    # MERGE THE RESULT WITH GROUPS INFORMATION
    merged = clients_groups.merge(loan_History_Evaluation, on='loan_id')
    return merged

# KABOOM.....KABOOM....KABOOOM.....
def merge2():
    df = merge1()
    df = df.sort_values(by='group_id')
    df_table = pd.pivot_table(data = df, index=['group_id'], values =['loan_id','days_overdue_2015','approved_principal','MaxOfOverdue 09_2015','MaxOfOverdue 10_2015','MaxOfOverdue 11_2015'],
                              aggfunc={'loan_id':'count','days_overdue_2015':np.mean, 'approved_principal':np.mean, 'MaxOfOverdue 09_2015':np.mean,
                              'MaxOfOverdue 10_2015':np.mean,'MaxOfOverdue 11_2015':np.mean})
    return df_table

def merge3():
    final_df = merge2()
    final_df['Past Three Months'] = final_df['MaxOfOverdue 11_2015'] + final_df['MaxOfOverdue 10_2015'] + final_df['MaxOfOverdue 09_2015']
    final_df['Past One Month'] = final_df['MaxOfOverdue 11_2015']
    final_df = final_df.rename(columns={'days_overdue_2015':'Average Over Due Days', 'loan_id':'Number Of Loans', 'approved_principal':'Average Principal'})
    final_df = final_df[['Number Of Loans','Average Principal','Past One Month','Past Three Months','Average Over Due Days']]
    return final_df

dataset = merge3()
count = dataset['Average Over Due Days'].count()
sum =  dataset['Average Over Due Days'].sum()
threshold = np.round((sum/count), 2) # Average of all days rounded to 2dp
print'Threshold : ', threshold
# 1 - High Risk (very risky)
# 0 - Low Risk(none risky)
dataset['Low/High Risk'] = np.where(dataset['Average Over Due Days'] < threshold, 0, 1)
dataset.to_csv('groups_final_dataset.csv', index=False)
print('---Done---')

