#!/usr/bin/env python3
# coding: utf-8

import glob
import os
import pickle
import random
from collections import Counter

import numpy as np
import math
import pandas as pd
import copy
from lomika_prep import *
from sklearn.preprocessing import StandardScaler
from util_lormika import *


class Explainer:

    # file initialization
    def __init__(self, **kwargs):
        self.__train_set = kwargs["train_set"]
        self.__train_class = kwargs["train_class"]
        self.__cases = kwargs["cases"]
        self.__model = kwargs["model"]
        self.__categorical_vars = kwargs["categorical_vars"]
        self.__continuous_vars = kwargs["continuous_vars"]
        self.__model_name = kwargs["model_name"]
        self.__path_data = kwargs["path_data"]
        self.__d_name = kwargs["d_name"]

    def make_nam_file_MO(self):
        target_train = self.__model.predict(self.__train_set)

        # class variables
        ori_dataset = pd.concat([self.__train_set.reset_index(drop=True), self.__train_class], axis=1)
        colnames = list(ori_dataset)  # Return column names
        nam_d = self.__d_name + '.nam'
        outF = self.__path_data + '/data_MO/defectsd/' + nam_d
        nam = open(outF, "w")

        for i in colnames:
            if i in self.__continuous_vars:
                line = i + ": numeric \n"
                nam.write(line)
            else:
                line = i + ": categorical \n"
                nam.write(line)

    def run_sequence(self):
        self.make_nam_file_MO()
        self.instance_generation()

    def instance_generation(self):

        rounds = 5
        # get the global model predictions for the training set
        target_train = self.__model.predict(self.__train_set)

        # class variables
        ori_dataset = pd.concat([self.__train_set.reset_index(drop=True), self.__train_class], axis=1)
        colnames = list(ori_dataset)  # Return column names

        # loop for five iterations
        for rd in range(1, rounds + 1):
            # Does feature scaling for continuous data and one hot encoding for categorical data
            scaler = StandardScaler()
            trainset_normalize = self.__train_set.copy()
            cases_normalize = self.__cases.copy()

            train_objs_num = len(trainset_normalize)
            dataset = pd.concat(objs=[trainset_normalize, cases_normalize], axis=0)
            dataset[self.__continuous_vars] = scaler.fit_transform(dataset[self.__continuous_vars])
            dataset = pd.get_dummies(dataset, prefix_sep="__", columns=self.__categorical_vars)
            trainset_normalize = copy.copy(dataset[:train_objs_num])
            cases_normalize = copy.copy(dataset[train_objs_num:])

            temp_df = pd.DataFrame([])

            # make dataframe to store similarities of the trained instances from the explained instance
            dist_df = pd.DataFrame(index=trainset_normalize.index.copy())

            test_num = 0
            width = math.sqrt(len(self.__train_set.columns)) * 0.75
            ####################################################### similarity
            for count, case in cases_normalize.iterrows():

                # Calculate the euclidian distance from the instance to be explained
                dist = np.linalg.norm(trainset_normalize.sub(np.array(case)), axis=1)

                # Convert distance to a similarity score
                similarity = np.sqrt(np.exp(-(dist ** 2) / (width ** 2)))

                dist_df['dist'] = similarity
                dist_df['t_target'] = target_train

                # get the unique classes of the training set
                unique_classes = dist_df.t_target.unique()
                # Sort similarity scores in to decending order
                dist_df.sort_values(by=['dist'], ascending=False, inplace=True)

                # Make a dataframe with top 40 elements in each class
                top_fourty_df = pd.DataFrame([])
                for clz in unique_classes:
                    top_fourty_df = top_fourty_df.append(dist_df[dist_df['t_target'] == clz].head(40))

                # get the minimum value of the top 40 elements and return the index
                cutoff_similarity = top_fourty_df.nsmallest(1, 'dist', keep='last').index.values.astype(int)[0]

                # Get the location for the given index with the minimum similarity
                min_loc = dist_df.index.get_loc(cutoff_similarity)
                # whole neighbourhood without undersampling the majority class
                train_neigh_sampling_b = dist_df.iloc[0:min_loc + 1]
                # get the size of neighbourhood for each class
                target_details = train_neigh_sampling_b.groupby(['t_target']).size()
                target_details_df = pd.DataFrame(
                    {'target': target_details.index, 'target_count': target_details.values})

                # Get the majority class and undersample
                final_neighbours_similarity_df = pd.DataFrame([])
                for index, row in target_details_df.iterrows():
                    if row["target_count"] > 200:
                        filterd_class_set = train_neigh_sampling_b.loc[
                            train_neigh_sampling_b['t_target'] == row['target']].sample(n=200)
                        final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)
                    else:
                        filterd_class_set = train_neigh_sampling_b.loc[
                            train_neigh_sampling_b['t_target'] == row['target']]
                        final_neighbours_similarity_df = final_neighbours_similarity_df.append(filterd_class_set)

                # Get the original training set instances which is equal to the index of the selected neighbours
                train_set_neigh = self.__train_set[self.__train_set.index.isin(final_neighbours_similarity_df.index)]
                print(train_set_neigh, "train set neigh")
                train_class_neigh = self.__train_class[
                    self.__train_class.index.isin(final_neighbours_similarity_df.index)]
                train_neigh_df = train_set_neigh.join(train_class_neigh)
                class_neigh = train_class_neigh.groupby(['target']).size()

                new_con_df = pd.DataFrame([])

                sample_classes_arr = []
                sample_indexes_list = []

                #######Generating 1000 instances using interpolation technique
                for num in range(0, 1000):
                    rand_rows = train_set_neigh.sample(2)
                    sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                    similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                    sample_classes = train_class_neigh[train_class_neigh.index.isin(rand_rows.index)]
                    sample_classes = np.array(sample_classes.to_records().view(type=np.matrix))
                    sample_classes_arr.append(sample_classes[0].tolist())

                    alpha_n = np.random.uniform(low=0, high=1.0)
                    x = rand_rows.iloc[0]
                    y = rand_rows.iloc[1]
                    new_ins = x + (y - x) * alpha_n
                    new_ins = new_ins.to_frame().T

                    # For Categorical Variables

                    for cat in self.__categorical_vars:

                        x_df = x.to_frame().T
                        y_df = y.to_frame().T

                        if similarity_both.iloc[0]['dist'] > similarity_both.iloc[1][
                            'dist']:  # Check similarity of x > similarity of y
                            new_ins[cat] = x_df.iloc[0][cat]
                        if similarity_both.iloc[0]['dist'] < similarity_both.iloc[1][
                            'dist']:  # Check similarity of y > similarity of x
                            new_ins[cat] = y_df.iloc[0][cat]
                        else:
                            new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat]])

                    new_ins.name = num
                    new_con_df = new_con_df.append(new_ins, ignore_index=True)

                #######Generating 1000 instances using cross-over technique
                for num in range(1000, 2000):
                    rand_rows = train_set_neigh.sample(3)
                    sample_indexes_list = sample_indexes_list + rand_rows.index.values.tolist()
                    similarity_both = dist_df[dist_df.index.isin(rand_rows.index)]
                    sample_classes = train_class_neigh[train_class_neigh.index.isin(rand_rows.index)]
                    sample_classes = np.array(sample_classes.to_records().view(type=np.matrix))
                    sample_classes_arr.append(sample_classes[0].tolist())

                    mu_f = np.random.uniform(low=0.5, high=1.0)
                    x = rand_rows.iloc[0]
                    y = rand_rows.iloc[1]
                    z = rand_rows.iloc[2]
                    new_ins = x + (y - z) * mu_f
                    new_ins = new_ins.to_frame().T

                    # For Categorical Variables get the value of the closest instance to the explained instance
                    for cat in self.__categorical_vars:
                        x_df = x.to_frame().T
                        y_df = y.to_frame().T
                        z_df = z.to_frame().T

                        new_ins[cat] = random.choice([x_df.iloc[0][cat], y_df.iloc[0][cat], z_df.iloc[0][cat]])

                    new_ins.name = num
                    new_con_df = new_con_df.append(new_ins, ignore_index=True)

                # get the global model predictions of the generated instances and the instances in the neighbourhood
                predict_dataset = train_set_neigh.append(new_con_df, ignore_index=True)
                target = self.__model.predict(predict_dataset)
                target_df = pd.DataFrame(target)

                neighbor_frequency = Counter(tuple(sorted(entry)) for entry in sample_classes_arr)

                new_df_case = pd.concat([predict_dataset, target_df], axis=1)
                new_df_case = np.round(new_df_case, 2)
                new_df_case.rename(columns={0: self.__train_class.columns[0]}, inplace=True)
                sampled_class_frequency = new_df_case.groupby(['target']).size()

                # Create dataMO file as the input to the Magnum Opus
                nameF = self.__path_data + '/data_MO/defectsd/' + self.__d_name + '_' + str(case.name) + '_' + str(
                    test_num) + '_' + str(
                    rd) + '_' + self.__model_name + '.data'
                np.savetxt(nameF, new_df_case.values, delimiter='@', newline='\n', fmt='%1.3f')
                test_num = test_num + 1


def models_pickle(path_data):
    model_load = []
    os.chdir(path_data + "/models")
    for file in glob.glob("*.pkl"):
        model_load.append(file)
    return model_load

def run_single_dataset():
    datsets_list = {
        'german': ('german_credit.csv', prepare_german_dataset),
        'adult': ('adult.csv', prepare_adult_dataset),
        'compas': ('compas-scores-two-years.csv', prepare_compass_dataset)
    }

    for dataset_kw in datsets_list:
        dataset_name, prepare_dataset_fn = datsets_list[dataset_kw]
        print(dataset_kw)
        if dataset_kw == 'adult':
            path_data = '/home/dilini/lomika/adult/'
        elif dataset_kw == 'compas':
            path_data = '/home/dilini/lomika/compass/'
        elif dataset_kw == 'german':
            path_data = '/home/dilini/lomika/german/'
        dataset = prepare_dataset_fn(dataset_name, path_data)
       # print(dataset, "dataset")
        train=dataset["train"]
        test=dataset["test"]
        numerical_vars=dataset["numerical_vars"]
        categorical_vars=dataset["categorical_vars"]
        model_load = models_pickle(path_data)
        for nb, black in enumerate(model_load):
            with open(path_data + '/models/' + black, 'rb') as f:
                blackbox = pickle.load(f)
            # print(blackbox)

            model_name = str(black).split(".")
            model_name = str(model_name[0])
            # print(model_name,"model_name")
            # test_rf_poor_correct, index_rf_poor_correct = get_test_sample(blackbox, test, poor=True, correct=True, seed=40,
            #                                                                           count=10)
            #test_cases = test[0:50]
            test_cases=test.iloc[: test.columns != 'target']
			model_kwargs = {
			                'train_set': train.loc[:, train.columns != 'target'],
			                'train_class': train[['target']],
			                'cases': test_cases,
			                'model': blackbox,
			                'categorical_vars': categorical_vars,
			                'continuous_vars': numerical_vars,
			                'model_name': model_name,
			                'path_data': path_data,
			                'd_name': dataset_kw,
            }
            exp = Explainer(**model_kwargs)
            exp.run_sequence()

def run_single_dataset():
    datasets_list = ['hive', 'groovy', 'activemq', 'derby', 'camel', 'lucene', 'hbase', 'wicket', 'jruby']

    for dataset_kw in datasets_list:
        path_data = '/home/defectsd/defects/ICSEnew/' + dataset_kw
        file_name = dataset_kw
        dataset = prepare_defect_dataset_fin(file_name, path_data)

        train = dataset["train"]
        test = dataset["test"]
        numerical_vars = dataset["numerical_vars"]
        categorical_vars = dataset["categorical_vars"]
        model_load = models_pickle(path_data)
        for nb, black in enumerate(model_load):
            with open(path_data + '/models/' + black, 'rb') as f:
                blackbox = pickle.load(f)

            model_name = str(black).split(".")
            model_name = str(model_name[0])
            test_cases = test.iloc[:, test.columns != 'target']

            model_kwargs = {
                'train_set': train.loc[:, train.columns != 'target'],
                'train_class': train[['target']],
                'cases': test_cases,
                'model': blackbox,
                'categorical_vars': categorical_vars,
                'continuous_vars': numerical_vars,
                'model_name': model_name,
                'path_data': path_data,
                'd_name': dataset_kw,
            }
            exp = Explainer(**model_kwargs)
            exp.run_sequence()


def main():
    run_single_dataset()
    print("Hello")

    # create object of the class


if __name__ == '__main__':
    main()
