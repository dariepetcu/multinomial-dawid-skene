import pandas as pd
import urllib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import noisy_annotators


# 0.005 = 6600 datapoints
# 0.0001 = 137 datapoints
def load_data(cleanup=False):
    # read in the data
    # comments = pd.read_csv('wiki_aggression/aggression_annotated_comments.tsv',  sep = '\t')
    if cleanup:
        annotations = pd.read_csv('wiki_aggression/aggression_annotations.tsv',  sep = '\t')
        # demographics = pd.read_csv('wiki_aggression/aggression_worker_demographics.tsv',  sep = '\t')

        # remove newline and tab tokens
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
        # comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

        # save datapoints with non-sparse annotations
        # start with list of all annotators
        all_worker_ids = annotations['worker_id'].unique().tolist()
        checked_ids = []
        min_annot = 7
        
        # Iterate through each rev_id
        print("Finding datapoints with reviews from all workers...")
        for rev_id in tqdm(annotations['rev_id'].unique()):
            end=False
            # Get worker_ids for the current rev_id
            rev_worker_ids = annotations[annotations['rev_id'] == rev_id]['worker_id'].unique().tolist()

            # Update the list of worker_ids by keeping only those present in rev_worker_ids
            for worker_id in all_worker_ids:
                if worker_id not in rev_worker_ids:
                    all_worker_ids.remove(worker_id)
            new_all_workers = [worker_id for worker_id in all_worker_ids if worker_id in rev_worker_ids]

            if new_all_workers:
                checked_ids.append(rev_id)
                all_worker_ids = new_all_workers

            # save on last iteration
            if annotations['rev_id'].unique()[-1] == rev_id:
                end = True

            if end:
                break


        # Select datapoints with reviews from remaining worker_ids
        selected_datapoints = annotations[annotations['rev_id'].isin(checked_ids)]
        # Concatenate selected datapoints to the new DataFrame
        print("Saving datapoints with reviews from all workers...")
        selected_datapoints.to_csv('filtered_datapoints.csv', index=False)

    else:
        selected_datapoints = pd.read_csv('filtered_datapoints.csv')

    # create test and train splits
    train, test = train_test_split(selected_datapoints, test_size=0.2, random_state=76)
    # selected_datapoints['split'] = selected_datapoints['rev_id'].apply(lambda x: 'train' if x % 5 != 0 else 'test')


    return train, test

# demographic can be 'all', or set of {'none', 'some', 'hs', 'bachelors', 'masters', 'doctorate', 'professional'}
def run_experiment(num_annotators=5, num_iterations=3, demographic_type='all', algorithm='multi'):
    train, test = load_data()
    match algorithm:
        case 'multi':
            print("Running multinomial EM")
            C_mats = noisy_annotators.EM_multinomial(train, num_iterations)
        case 'dawid':
            print("Running Dawid-Skene EM")
            C_mats = noisy_annotators.EM_dawid_skene(train, num_iterations)




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