# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 11:07:34 2016

@author: AGGREY
"""
import pandas as pd

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


#Load Loans Information
def loans_information():
    file1_name = 'musoni/m_loan.csv'
    required_columns = [0,3,7,15,16,34,36] 
    LoansData = LoadDataset().load_file(file1_name,required_columns)
    LoansData.rename(columns={'id':'loan_id'}, inplace=True)
    return LoansData;

#Load clients information
def clients_information():
    file2_name = 'musoni/clients.csv'
    required_columns = [0,5,14,16,17]
    clientsData = LoadDataset().load_file(file2_name, required_columns)
    #Name the id->client_id for merging purposes
    clientsData.rename(columns ={'id':'client_id', 'activation_date':'client_activation_date', 'gender_cv_id': 'gender'}, inplace=True)
    return clientsData

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

# Join the Loans and Clients DataFrames
def loans_clients(loans, clients):
    Loans_Clients_Merged = loans.merge(clients, on='client_id')
    return Loans_Clients_Merged


#Join Loans Officers and the Offices they are attached to
def offices_officeLocation(loan_officers, office_locations):
    merged = loan_officers.merge(office_locations, on='office_id')
    return merged


# Merge Loans_Clients_Merged AND LoansOfficers_OfficesAttached_merged
def final_dataset(loans_clients_df, staff_and_offices_df):
    finalDataset = loans_clients_df.merge(staff_and_offices_df, on='loan_officer_id')
    #change the order in which the columns appear
    finalDataset = finalDataset[['client_id','display_name', 'gender','date_of_birth','client_activation_date','loan_id','principal_amount','approved_principal','submittedon_date','approvedon_date','loan_officer_id','loan_officer_name','office_id','office_name']]
    finalDataset.to_csv('loans_clients_officers_location_combined.csv', index=False)
    print('Finished Saving to the CSV File')


loans = loans_information()
clients = clients_information()
loans_clients_df = loans_clients(loans, clients)
loan_officers = loans_officers()
office_locations = office_locations()
staff_and_offices_df = offices_officeLocation(loan_officers, office_locations)
final_dataset(loans_clients_df,staff_and_offices_df)












