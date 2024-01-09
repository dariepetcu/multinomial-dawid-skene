import pandas as pd
import urllib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle, random
import noisy_annotators


# 0.005 = 6600 datapoints
# 0.0001 = 137 datapoints
def load_data(num_datapoints=10000, cleanup=False):
    # read in the data
    comments = pd.read_csv('wiki_aggression/aggression_annotations.tsv',  sep = '\t')
    # select first 1000 rows of comments
    
    filename = f'non_sparse_{num_datapoints}.csv'
    if cleanup:
        # keep clean subset of available data
        selected_datapoints = handle_sparsity(comments.head(num_datapoints))
        print("Generating ground truths...")
        annotated = generate_labels(selected_datapoints)
        # Concatenate selected datapoints to the new DataFrame
        print("Saving datapoints with selected reviews...")
        annotated.to_csv(filename, index=False)
    else:
        annotated = pd.read_csv(filename)


    # create test and train splits
    train, test = train_test_split(annotated, test_size=0.2, random_state=76)

    return train, test


def handle_sparsity(annotations, no_revs=10):
    reviews = annotations['rev_id'].unique().tolist()
    all_worker_ids = annotations['worker_id'].unique().tolist()

    final_reviews = set()
    final_annotators = set()

    # pickled_annot_by_rev = 'good_annots_by_rev.pkl'
    pickled_rev_by_annot = 'good_rev_by_annot.pkl'


    with open(pickled_rev_by_annot, 'rb') as f:
        rev_by_annot = pickle.load(f)
        print("Unpickled rev by annot.")

    # keep only annotators who annotated more than 1000 reviews
    for annot in rev_by_annot.keys():
        if len(rev_by_annot[annot]) > no_revs:
            final_annotators.add(annot)
            final_reviews.update(rev_by_annot[annot])

    # with open(pickled_annot_by_rev, 'rb') as f:
    #     annot_by_rev = pickle.load(f)
    #     print("Unpickled data.")
    print("finished parsing data, removing reviews from deleted annotators..")

    # Keep subset of dataframe with selected reviews
    dense_df = annotations[annotations['rev_id'].isin(final_reviews)]
    dense_df = dense_df[dense_df['worker_id'].isin(final_annotators)]

    print(f"New dataframe contains {len(dense_df)} datapoints")

    return dense_df


def generate_labels(annotations):
    reviews = annotations['rev_id'].unique().tolist()
    annotators = annotations['worker_id'].unique().tolist()

    # create columns for the ground truths
    annotations['MV Truth'] = None
    annotations['DS Truth'] = None
    
    # train DS on whole dataset to generate labels
    DS_mats = noisy_annotators.EM_Skeene(annotations)

    # add labels to dataset
    for review in reviews:

        # majority voting
        num_annotations = len(annotations[annotations['rev_id']==review])
        num_true = len(annotations[(annotations['aggression']==1) & (annotations['rev_id']==review)])
        num_true /= num_annotations
        maj_voting = 1 if num_true >= 0.5 else 0
        # add MV ground truth to review in dataframe
        annotations.loc[annotations['rev_id']==review, 'MV Truth'] = maj_voting

        # DS sampling
        for annotator in annotators:
            # make sure datapoint exists
            datapoints = annotations[(annotations['worker_id']==annotator) & (annotations['rev_id']==review)]
            if len(datapoints) > 0:
                # check DS label against prediction
                # label = check_matrix_truth(datapoints.iloc[0]['aggression'], DS_mats[annotator])
                prediction = datapoints.iloc[0]['aggression']
                label = check_matrix_truth(prediction, DS_mats[annotator])
                # add DS ground truth to review in dataframe
                annotations.loc[(annotations['worker_id']==annotator) & (annotations['rev_id']==review), 'DS Truth'] = label

    return annotations


def check_matrix_truth(prediction, C_mat):
    sample = random.uniform(0,1)
    if prediction == 0:
        label = 0 if C_mat[0][0] > sample else 1
    else:
        label = 1 if C_mat[1][1] > sample else 0

    return label


# demographic can be 'all', or set of {'none', 'some', 'hs', 'bachelors', 'masters', 'doctorate', 'professional'}
def run_experiment(num_datapoints=10000, num_iterations=5, algorithm='ds', cleanup=False):
    train, test = load_data(num_datapoints, cleanup)

    # train algorithm on training set
    match algorithm:
        case 'mn':
            print("Running multinomial EM")
            C_mats = noisy_annotators.EM_multinomial(train, num_iterations)
        case 'ds':
            print("Running Dawid-Skene EM")
            C_mats = noisy_annotators.EM_Skeene(train, num_iterations)


    # use matrices to test on test set
            
    DS_correct = 0
    MV_correct = 0
    DS_missing = 0

    print("Starting test phase")
    # test_set = test['rev_id'].unique()
    for _, datapoint in tqdm(test.iterrows()):
        # get C_mat for annotator
        match algorithm:
            case 'mn':
                C_mat = C_mats
            case 'ds':
                C_mat = C_mats.get(datapoint['worker_id'])
                # it is possible that the annotator was not in the training set
                if C_mat is None:
                    DS_missing += 1
                    continue
        # get prediction for datapoint
        prediction = check_matrix_truth(datapoint['aggression'], C_mat)
        
        # prediction is correct
        if datapoint['DS Truth'] == prediction:
            DS_correct += 1
        
        if datapoint['MV Truth'] == prediction:
            MV_correct += 1

    print(f"DS accuracy: {DS_correct/(len(test) - DS_missing)}")
    print(f"MV accuracy: {MV_correct/len(test)}")
            




def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapoints", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--algo", type=str, default='mn')
    parser.add_argument("--parse_data", action="store_true", default=False)

    args = parser.parse_args()
    run_experiment(num_datapoints=args.datapoints, num_iterations=args.iterations, algorithm=args.algo, cleanup=args.parse_data)



if __name__ == "__main__":
    main()


# pickled_annot_by_rev = 'good_annots_by_rev.pkl'
# pickled_rev_by_annot = 'good_rev_by_annot.pkl'


# with open(pickled_rev_by_annot, 'rb') as f:
#     rev_by_annot = pickle.load(f)
#     print("Unpickled data.")

# with open(pickled_annot_by_rev, 'rb') as f:
#     annot_by_rev = pickle.load(f)
#     print("Unpickled data.")

# for i, rev in enumerate(annot_by_rev.keys()):
#     print("rev id: ", rev)

#     if i > 100:
#         break




##############################################################################################################################
# comments structure:
#    rev_id                                            comment  year  logged_in       ns  sample  split
# 0   37675  `- This is not ``creative``.  Those are the di...  2002       True  article  random  train

# annotations structure:
#    rev_id  worker_id  aggression  aggression_score
# 0   37675       1362         1.0              -1.0

# demographics structure:
#    worker_id  gender  english_first_language age_group     education
# 0        833  female                       0     45-60     bachelors

##############################################################################################################################

# Schema for aggression_annotations.tsv
# Aggression labels from several crowd-workers for each comment in aggression_annotated_comments.tsv. It can be joined with aggression_annotated_comments.tsv on rev_id.

# rev_id: MediaWiki revision id of the edit that added the comment to a talk page (i.e. discussion).
# worker_id: Anonymized crowd-worker id.
# aggression_score: Categorical variable ranging from very aggressive (-3), to neutral (0), to very friendly (3).
# aggression: Indicator variable for whether the worker thought the comment has an aggressive tone . The annotation takes on the value 1 if the worker considered
# the comment aggressive (i.e worker gave an aggression_score less than 0) and value 0 if the worker considered the comment neutral or friendly (i.e worker gave
# an aggression_score greater or equal to 0). Takes on values in {0, 1}.

##############################################################################################################################

# Schema for {attack/aggression/toxicity}_worker_demographics.tsv
# Demographic information about the crowdworkers. This information was obtained by an optional demographic survey administered after the labelling task.
# It is meant to be joined with {attack/aggression/toxicity}_annotations.tsv on worker_id. Some fields may be blank if left unanswered.

# worker_id: Anonymized crowd-worker id.
# gender: The gender of the crowd-worker. Takes a value in {'male', 'female', and 'other'}.
# english_first_language: Does the crowd-worker describe English as their first language. Takes a value in {0, 1}.
# age_group: The age group of the crowd-worker. Takes on values in {'Under 18', '18-30', '30-45', '45-60', 'Over 60'}.
# education: The highest education level obtained by the crowd-worker. Takes on values in {'none', 'some', 'hs', 'bachelors', 'masters', 'doctorate', 'professional'}.
# Here 'none' means no schooling, some means 'some schooling', 'hs' means high school completion, and the remaining terms indicate completion of
# the corresponding degree type.








# labels a comment as an atack if the majority of annoatators did so
# labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
# join labels and comments
# comments['attack'] = labels
# # remove newline and tab tokens
# comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
# comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
# comments.query('attack')['comment'].head()
# # fit a simple text classifier

# train_comments = comments.query("split=='train'")
# test_comments = comments.query("split=='test'")

# clf = Pipeline([
#     ('vect', CountVectorizer(max_features = 10000, ngram_range = (1,2))),
#     ('tfidf', TfidfTransformer(norm = 'l2')),
#     ('clf', LogisticRegression()),
# ])
# clf = clf.fit(train_comments['comment'], train_comments['attack'])
# auc = roc_auc_score(test_comments['attack'], clf.predict_proba(test_comments['comment'])[:, 1])
# print('Test ROC AUC: %.3f' %auc)

# # correctly classify nice comment
# clf.predict(['Thanks for you contribution, you did a great job!'])

# # correctly classify nasty comment
# clf.predict(['People as stupid as you should not edit Wikipedia!'])