import linktransformer as lt
import pandas as pd
import pickle

def calculate_retrieval_accuracy(path_to_pickle):
    with open(path_to_pickle, 'rb') as handle:
        val_data = pickle.load(handle)

    # Merge queries and corpus using LinkTransformer's lt.merge function
    merged_df = lt.merge(pd.DataFrame.from_dict(val_data[0], orient='index', columns=['query_text']).reset_index(), 
                         pd.DataFrame.from_dict(val_data[1], orient='index', columns=['corpus_text']).reset_index(), 
                         merge_type='1:m', model="hiiamsid/sentence_similarity_spanish_es", 
                         left_on="query_text", right_on="corpus_text", openai_key=None)

    # Create a new column in merged_df that indicates whether each document (cid) is relevant to its corresponding query (qid)
    def is_relevant(row):
        return 1 if row['cid'] in val_data[2][row['qid']] else 0

    merged_df['is_relevant'] = merged_df.apply(is_relevant, axis=1)

    # Calculate the retrieval accuracy by taking the mean of the is_relevant column
    accuracy = merged_df['is_relevant'].mean()

    return accuracy


