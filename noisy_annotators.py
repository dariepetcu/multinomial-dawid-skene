import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import pickle



def get_prior_over_t(dataset):
    count_t1 = len(dataset[dataset['aggression']==1.0])
    return count_t1/len(dataset)


def EM_multinomial(dataset, num_iterations=3):
    # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    C_mat = np.random.rand(2,2)
    datapoints = dataset['rev_id'].unique()
    data_size = len(dataset)

    p_hat = np.random.rand(len(datapoints), 2)
    prior_over_t = get_prior_over_t(dataset)
    print("prior_over_t: ", prior_over_t)
    
    # create lookup between datapoint id and index in p_hat
    datapoints = dataset['rev_id'].unique()
    id_to_idx = {elem:i for i, elem in enumerate(datapoints)}
    # pickle the lookup
    lookup_pickle = f'mn_id_to_idx_{data_size}.pkl'
    with open(lookup_pickle, 'wb') as f:
        pickle.dump(id_to_idx, f)
        
    classes = [0, 1]

    for iteration_idx in tqdm(range(num_iterations)):
        print("E step started")
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
        # pickle p_hat
        p_hat_file = f'mn_{data_size}_p_hat_epoch{iteration_idx}.pkl'
        with open(p_hat_file, 'wb') as f:
            pickle.dump(p_hat, f)
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
        # pickle C_mat
        c_mat_file = f'mn_{data_size}_c_mat_epoch{iteration_idx}.pkl'
        with open(c_mat_file, 'wb') as f:
            pickle.dump(C_mat, f)
    # # return inside dict to match dtype of other EM functions
    # C_mats = {' ': C_mat}
    return C_mat


def EM_Skeene(dataset, num_iterations=3):
    datapoints = dataset['rev_id'].unique()
    # store all C_mat in a dictionary with annotator ids as key
    annotators = dataset['worker_id'].unique()
    annotator_matrices = {int(annotator): np.random.rand(2,2) for annotator in annotators}

    # initialize p_hat to random values
    p_hat = np.random.rand(len(datapoints), 2)
    prior_over_t = get_prior_over_t(dataset)
    print("prior_over_t: ", prior_over_t)
    data_size = len(dataset)
    # create lookup between datapoint id and index in p_hat
    datapoints = dataset['rev_id'].unique()
    id_to_idx = {elem:i for i, elem in enumerate(datapoints)}
    # pickle the lookup
    lookup_pickle = f'id_to_idx_{data_size}.pkl'
    with open(lookup_pickle, 'wb') as f:
        pickle.dump(id_to_idx, f)
    # classes = [0.0, 1.0]
    classes = [0, 1]

    for iteration_idx in range(num_iterations):

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
        # pickle p_hat
        p_hat_file = f'ds_{data_size}_p_hat_epoch{iteration_idx}.pkl'
        with open(p_hat_file, 'wb') as f:
            pickle.dump(p_hat, f)
        # M-step
        for l in tqdm(annotators):
            C_mat = annotator_matrices[l]
            for k in classes:
                for j in classes:
                    # compute equation 25
                    sum_numerator = 0
                    sum_denominator = 0
                    for inner_j in classes:
                        for n in datapoints:
                            y_nlj = len(dataset[(dataset['aggression']==j) & (dataset['rev_id']==n) & (dataset['worker_id']==l)])
                            y_nlj = min(y_nlj, 1)
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
        # pickle C_mat
        c_mat_file = f'ds_{data_size}_c_mat_epoch{iteration_idx}.pkl'
        with open(c_mat_file, 'wb') as f:
            pickle.dump(annotator_matrices, f)
    return annotator_matrices