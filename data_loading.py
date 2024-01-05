import pandas as pd
import urllib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# import noisy_annotators


# 0.005 = 6600 datapoints
# 0.0001 = 137 datapoints
def load_data(cleanup=False):
    # read in the data
    comments = pd.read_csv('wiki_aggression/aggression_annotations.tsv',  sep = '\t')
    # select first 1000 rows of comments
    if cleanup:
        # keep clean subset of available data
        selected_datapoints = handle_sparsity(comments, no_anots=2)      
        # Concatenate selected datapoints to the new DataFrame
        print("Saving datapoints with reviews from all workers...")
        selected_datapoints.to_csv('non_sparse_try1000.csv', index=False)
    else:
        selected_datapoints = pd.read_csv('non_sparse.csv')

    # generate_labels(selected_datapoints)

    # create test and train splits
    # train, test = train_test_split(selected_datapoints, test_size=0.2, random_state=76)

    # return train, test


def handle_sparsity(annotations, no_anots=7):
    reviews = annotations['rev_id'].unique().tolist()
    all_worker_ids = annotations['worker_id'].unique().tolist()

    # create dict of reviews by each annotator
    reviews_by_annot = {}
    for annotator in tqdm(all_worker_ids):
        submitted_reviews = set(annotations[annotations['worker_id'] == annotator]['rev_id'].unique())
        reviews_by_annot[annotator] = submitted_reviews

    # sort annotators by number of reviews submitted
    best_annotators = sorted(reviews_by_annot, key=lambda key: len(reviews_by_annot[key]))
    # start with all reviews annotated by the best annotator. store review ids
    final_reviews = set()

    # keep best 7 annotators
    picked_annots = best_annotators[-no_anots:]
    # print("picked annotators: ", picked_annots)
    for rev in tqdm(reviews):
        ok = True
        for annot in picked_annots:
            # remove datapoint if one of top 7 annotators did not annotate it
            if rev not in reviews_by_annot[annot]:
                # stop checking current datapoint
                ok = False
                break
        if ok:
            final_reviews.add(rev)


    # store worker ids
    # picked_annots = all_worker_ids

    # # iterate through annotators to create non-sparse dataset. end condition is going below selected number of annotators.
    # for annot in tqdm(best_annotators[1:]):
    #     if len(picked_annots) < no_anots:
    #         break
    #     # get reviews annotated by current annotator
    #     annot_reviews = reviews_by_annot[annot]
    #     # get reviews annotated by previous annotators but not by current annotator
    #     currently_unavailable = picked_reviews.difference(annot_reviews)
    #     # decide whether to drop some datapoints or skip this annotator
    #     if len(currently_unavailable) < picked_reviews/shrinkage_ratio:
    #         # drop some datapoints
    #         picked_reviews = picked_reviews.intersection(annot_reviews)
    #     else:
    #         # skip this annotator
    #         picked_annots.remove(annot)

    print("finished parsing data, removing reviews from deleted annotators..")

    # Keep subset of dataframe with selected reviews
    dense_df = annotations[annotations['rev_id'].isin(final_reviews)]
    dense_df = dense_df[dense_df['worker_id'].isin(picked_annots)]

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
        # TODO add column to dataframe

        # DS sampling
        for annotator in annotators:
            # make sure datapoint exists
            if len(annotations[(annotations['worker_id']==annotator) & (annotations['rev_id']==review)]) > 0:
                # get matrix
                C_mat = DS_mats[annotator]
                label = 1 if C_mat[0][0] < C_mat[1][1] else 0
                # TODO add column to dataframe
    
    return labeled_dataframe




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


load_data(cleanup=True)

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