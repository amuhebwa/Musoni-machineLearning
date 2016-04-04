# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 11:07:34 2016

@author: AGGREY
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#Load dataset
class LoadDataset(object):
    """Loads dataset from CSV file"""
    
    def __init__(self):
        pass
    
    def load_file(self, file_name, required_columns=None):
        if(required_columns is None):
            dataset = pd.read_csv(file_name, encoding='ISO-8859-2', low_memory=False)
        else:
            dataset = pd.read_csv(file_name, encoding='ISO-8859-2', usecols=required_columns, low_memory=False)       
        return dataset


#Load Loans Information
file1_name = 'musoni/m_loan.csv'
required_columns = [3,7,15,16,34,36] 
LoansData = LoadDataset().load_file(file1_name,required_columns)

#Load clients information
file2_name = 'musoni/clients.csv'
required_columns = [0,5,16,17]
clientsData = LoadDataset().load_file(file2_name, required_columns)
#Name the id->client_id for merging purposes
clientsData.rename(columns ={'id':'client_id', 'activation_date':'client_activation_date', 'gender_cv_id': 'gender'}, inplace=True)

#Load Loan Officers' details
file3_name = 'musoni/loan_officers.csv'
required_columns=[0,2,7]
loansOfficers = LoadDataset().load_file(file3_name, required_columns)
loansOfficers.rename(columns = {'id' : 'loan_officer_id','external_id' : 'loan_officer_name'}, inplace=True)


# Load Office Locations
file4_name = 'musoni/m_office.csv'
required_columns = [0,4]
officeLocations = LoadDataset().load_file(file4_name, required_columns)
officeLocations.rename(columns={'id':'office_id', 'name':'office_Name'}, inplace=True)

# Join the Loans and Clients DataFrames
Loans_Clients_Merged = LoansData.merge(clientsData, on='client_id')

#Join Loans Officers and the Offices they are attached to
LoansOfficers_OfficesAttached_merged = loansOfficers.merge(officeLocations, on='office_id')


# Merge Loans_Clients_Merged AND LoansOfficers_OfficesAttached_merged
finalDataset = Loans_Clients_Merged.merge(LoansOfficers_OfficesAttached_merged, on='loan_officer_id')

#change the order in which the columns appear
finalDataset = finalDataset[['client_id','gender','date_of_birth','client_activation_date',
'principal_amount','approved_principal','submittedon_date','approvedon_date','loan_officer_id',
'loan_officer_name','office_id','office_Name']]

finalDataset.to_csv('aggrey_muhebwa.csv', index=False)
print('Finished Saving to the CSV File')







