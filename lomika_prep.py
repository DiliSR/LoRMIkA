#!/usr/bin/env python3
# coding: utf-8


import pickle
import pandas as pd

from util_lomika import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def prepare_adult_dataset(filename, path_data):
    ## 0 - poor    1 - rich
    # Read dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True, engine='python')
    df.rename(columns={'class': 'target', 'education-num': 'education_num', 'marital-status': 'marital_status',
                       'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss',
                       'hours-per-week': 'hours_per_week', 'native-country': 'native_country'}, inplace=True)
    # Remove useless columns
    del df['fnlwgt']
    del df['education_num']

    # Remove Missing Values
    for col in df.columns:
        if '?' in df[col].unique():
            df[col][df[col] == '?'] = df[col].value_counts().index[0]

    df_bl = df.copy()
    label_le = LabelEncoder()
    df['target'] = label_le.fit_transform(df['target'].values)

    # Numerical variables
    numerical_vars = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

    # Categorical variables
    categorical_vars = ['workclass','marital_status', 'occupation', 'education',
                        'relationship', 'race', 'sex', 'native_country']

    df_le, label_encoder = label_encode(df, categorical_vars, label_encoder=None)

    df_le.to_csv('/home/dilini/lomika/adult/data_MO/adult.csv', sep=',', encoding='utf-8')
    train, test = train_test_split(df_le, test_size=0.2, random_state=0)

    dataset = {
        'name': filename.replace('.csv', ''),
        'df_bl': df_bl,
        'categorical_vars': categorical_vars,
        'numerical_vars': numerical_vars,
        'label_encoder': label_encoder,
        'train': train,
        'test': test,
        'df_le': df_le
    }
    return dataset


def prepare_german_dataset(filename, path_data):
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',')

    df.rename(columns={'default': 'target'}, inplace=True)

    
    label_le = LabelEncoder()
    df['target'] = label_le.fit_transform(df['target'].values)

    numerical_vars = ['duration_in_month', 'credit_amount', 'age']
    categorical_vars = ['other_debtors', 'people_under_maintenance', 'personal_status_sex', 'present_res_since',
                        'credits_this_bank', 'account_check_status', 'job', 'savings', 'credit_history',
                        'purpose', 'present_emp_since', 'property', 'other_installment_plans', 'housing',
                        'installment_as_income_perc', 'telephone', 'foreign_worker']


    df_le, label_encoder = label_encode(df, categorical_vars, label_encoder=None)
    if 'Unnamed: 0' in df_le.columns:
        df_le.drop('Unnamed: 0', axis=1, inplace=True)

    df_le.to_csv('/home/dilini/lomika/german/data_MO/german.csv',sep=',', encoding='utf-8')
    if 'Unnamed: 0' in df_le.columns:
        df_le.drop('Unnamed: 0', axis=1, inplace=True)
    df_bl = df_le.copy()
    train, test = train_test_split(df_le, test_size=0.2, random_state=0)

    dataset = {
        'name': filename.replace('.csv', ''),
        'df_bl': df_bl,
        'categorical_vars': categorical_vars,
        'numerical_vars': numerical_vars,
        'label_encoder': label_encoder,
        'train': train,
        'test': test,
        'df_le': df_le
    }
    return dataset


def prepare_compass_dataset(filename, path_data):
    # Read Dataset
    df = pd.read_csv(path_data + filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
    df.rename(columns={'age_cat': 'birthgroup'}, inplace=True)
    print(df)
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    print(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype('int64')
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype('int64')
    df['is_recid'] = df['is_recid'].apply(str)
    df['is_violent_recid'] = df['is_violent_recid'].apply(str)
    df['two_year_recid'] = df['two_year_recid'].apply(str)
    df['is_recid'] = df['is_recid'].astype(object)
    df['is_violent_recid'] = df['is_violent_recid'].astype(object)
    df['two_year_recid'] = df['two_year_recid'].astype(object)

    def get_class(x):
        if x < 7:
            return 'Medium_Low'
        else:
            return 'High'

    df['target'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    df_bl = df.copy()
    label_le = LabelEncoder()
    df['target'] = label_le.fit_transform(df['target'].values)

    numerical_vars = ['age', 'priors_count', 'days_b_screening_arrest', 'length_of_stay']
    categorical_vars = ['is_recid',
                        'c_charge_degree',
                        'is_violent_recid',
                        'birthgroup',
                        'two_year_recid',
                        'race',
                        'sex']

    
    df_le, label_encoder = label_encode(df, categorical_vars, label_encoder=None)
    if 'Unnamed: 0' in df_le.columns:
        df_le.drop('Unnamed: 0', axis=1, inplace=True)
    df_le.to_csv('/home/dilini/lomika/compass/data_MO/compass.csv',sep=',', encoding='utf-8')
    if 'Unnamed: 0' in df_le.columns:
        df_le.drop('Unnamed: 0', axis=1, inplace=True)
    df_bl = df_le.copy()
    train, test = train_test_split(df_le, test_size=0.2, random_state=0)

    dataset = {
        'name': filename.replace('.csv', ''),
        'df_bl': df_bl,
        'categorical_vars': categorical_vars,
        'numerical_vars': numerical_vars,
        'label_encoder': label_encoder,
        'train': train,
        'test': test,
        'df_le': df_le
    }
    return dataset

