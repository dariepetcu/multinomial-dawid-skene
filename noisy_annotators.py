import numpy as np
import random
import pandas as pd
import data_loading
from tqdm import tqdm



def get_prior_over_t(dataset):
    count_t1 = len(dataset[dataset['aggression']==1.0])
    return count_t1/len(dataset)


def EM_multinomial(dataset, num_iterations):
    # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    C_mat = np.random.rand(2,2)
    datapoints = dataset['rev_id'].unique()

    p_hat = np.random.rand(len(datapoints), 2)
    prior_over_t = get_prior_over_t(dataset)
    print("prior_over_t: ", prior_over_t)
    
    # create lookup between datapoint id and index in p_hat
    datapoints = dataset['rev_id'].unique()
    id_to_idx = {elem:i for i, elem in enumerate(datapoints)}
    # classes = [0.0, 1.0]
    classes = [0, 1]

    for _ in tqdm(range(num_iterations)):

        # E-step
        for n in datapoints:
            for k in classes:
                # compute equation 13
                prod_numerator = 1
                sum_denominator = 0
                for i in classes:
                    prod_denominator = 1
                    for j in classes:
                        # how many annotators label datapoint n as class j
                        sum_annotators = len(dataset[(dataset['aggression']==j) & (dataset['rev_id']==n)])
                        # compute pi^sum_annotators
                        current_pi = np.power(C_mat[i][j], sum_annotators)
                        # multiply by class prior
                        prod_denominator *= current_pi
                        # compute numerator of equation 13
                        if i == k:
                            prod_numerator *= current_pi
                    if i > 0:
                        sum_denominator += prod_denominator * prior_over_t
                    else:
                        sum_denominator += prod_denominator * (1 - prior_over_t)
                # multiply numerator by class prior
                if k > 0:                    
                    prod_numerator *= prior_over_t
                else:
                    prod_numerator *= (1 - prior_over_t)

                # update p_hat
                p_hat[id_to_idx[n]][k] = prod_numerator / sum_denominator
                # print("n: ", n, "k: ", k, "phat ", p_hat[id_to_idx[n]][k])

        print("E step complete")
        # M-step
        for k in classes:
            for j in classes:
                # compute equation 15
                sum_numerator = 0
                sum_denominator = 0
                for inner_j in classes:
                    for n in datapoints:
                        current_term = p_hat[id_to_idx[n]][k] * \
                                len(dataset[(dataset['aggression']==inner_j) & (dataset['rev_id']==n)])
                        sum_denominator += current_term
                        # numerator is one element of the sum in denominator
                        if inner_j == j:
                            sum_numerator += current_term
                                
                C_mat[k][j] = sum_numerator / sum_denominator

    # return inside dict to match dtype of other EM functions
    C_mats = {' ': C_mat}
    return C_mats


def EM_Skeene(dataset, num_iterations):
    datapoints = dataset['rev_id'].unique()
    # store all C_mat in a dictionary with annotator ids as key
    annotators = dataset['worker_id'].unique()
    annotator_matrices = {annotator: np.random.rand(2,2) for annotator in annotators}

    # initialize p_hat to random values
    p_hat = np.random.rand(len(datapoints), 2)
    prior_over_t = get_prior_over_t(dataset)
    print("prior_over_t: ", prior_over_t)
    
    # create lookup between datapoint id and index in p_hat
    datapoints = dataset['rev_id'].unique()
    id_to_idx = {elem:i for i, elem in enumerate(datapoints)}
    # classes = [0.0, 1.0]
    classes = [0, 1]

    for _ in range(num_iterations):

        # E-step
        for n in tqdm(datapoints):
            for k in classes:
                # compute equation 23
                prod_numerator = 1
                sum_denominator = 0
                for i in classes:
                    prod_denominator = 1
                    for l in annotators:
                        for j in classes:
                            # how many annotators with id l label datapoint n as class j
                            y_nlj = len(dataset[(dataset['aggression']==j) & (dataset['rev_id']==n) & (dataset['worker_id']==l)])
                            # compute pi^y_nlj
                            C_mat = annotator_matrices[l]
                            current_pi = np.power(C_mat[i][j], y_nlj)
                            # multiply by class prior
                            prod_denominator *= current_pi
                            # compute numerator of equation 13
                            if i == k:
                                prod_numerator *= current_pi
                        if i > 0:
                            sum_denominator += prod_denominator * prior_over_t
                        else:
                            sum_denominator += prod_denominator * (1 - prior_over_t)
                # multiply numerator by class prior
                if k > 0:                    
                    prod_numerator *= prior_over_t
                else:
                    prod_numerator *= (1 - prior_over_t)

                # update p_hat
                p_hat[id_to_idx[n]][k] = prod_numerator / sum_denominator
                # print("n: ", n, "k: ", k, "phat ", p_hat[id_to_idx[n]][k])

        print("E step complete")
        # M-step
        for l in annotators:
            C_mat = annotator_matrices[l]
            for k in classes:
                for j in classes:
                    # compute equation 25
                    sum_numerator = 0
                    sum_denominator = 0
                    for inner_j in classes:
                        for n in datapoints:
                            #TODO prevent y_nlj from always being 0 in some cases
                            y_nlj = len(dataset[(dataset['aggression']==j) & (dataset['rev_id']==n) & (dataset['worker_id']==l)])
                            current_term = p_hat[id_to_idx[n]][k] * y_nlj
                            sum_denominator += current_term
                            # numerator is one element of the sum in denominator
                            if inner_j == j:
                                sum_numerator += current_term
                    # update C_mat
                    # prevent division by 0
                    sum_denominator = max(sum_denominator, 1)
                    C_mat[k][j] = sum_numerator / sum_denominator
                    # update matrix in the dictionary
                    annotator_matrices[l] = C_mat

    return annotator_matrices


annotations = data_loading.load_data(cleanup=True) 
# C_mat = EM_multinomial(annotations, 1)
# print(C_mat)

annotator_matrices = EM_Skeene(annotations, 10)
for i, annotator in enumerate(annotator_matrices.keys()):
    print(annotator)
    print(annotator_matrices[annotator])
    if i == 10:
        break