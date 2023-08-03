import os
import sys
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import evaluation, LoggingHandler
import logging
from transformers import logging as lg
import wandb
from typing import List, Dict, Set, Callable
from torch import Tensor

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from tqdm.autonotebook import trange
import torch
import heapq

lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)



def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


class InformationRetrievalEvaluator_wandb(evaluation.InformationRetrievalEvaluator):
    """Information Retrieval """
    def __init__(self,
                 queries: Dict[str, str],  #qid => query
                 corpus: Dict[str, str],  #cid => doc
                 relevant_docs: Dict[str, Set[str]],  #qid => Set[cid]
                 corpus_chunk_size: int = 50000,
                 mrr_at_k: List[int] = [10],
                 ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10],
                 precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [100],
                 show_progress_bar: bool = False,
                 batch_size: int = 32,
                 name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor] ] = {'cos_sim': cos_sim},       #Score function, higher=more similar
                 main_score_function: str = None,
                 wandb_names=None
                 ):

        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)
        self.queries_raw= queries
        self.corpus_raw= corpus

        self.queries = [queries[qid] for qid in self.queries_ids]

        self.corpus_ids = list(corpus.keys())
        self.corpus = [corpus[cid] for cid in self.corpus_ids]

        self.relevant_docs = relevant_docs
        self.corpus_chunk_size = corpus_chunk_size
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys()))
        self.main_score_function = main_score_function
        self.wandb_names = wandb_names

        if name:
            name = "_" + name

        self.csv_file: str = "Information-Retrieval_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        for score_name in self.score_function_names:
            for k in accuracy_at_k:
                self.csv_headers.append("{}-Accuracy@{}".format(score_name, k))

            for k in precision_recall_at_k:
                self.csv_headers.append("{}-Precision@{}".format(score_name, k))
                self.csv_headers.append("{}-Recall@{}".format(score_name, k))

            for k in mrr_at_k:
                self.csv_headers.append("{}-MRR@{}".format(score_name, k))

            for k in ndcg_at_k:
                self.csv_headers.append("{}-NDCG@{}".format(score_name, k))

            for k in map_at_k:
                self.csv_headers.append("{}-MAP@{}".format(score_name, k))


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = " after epoch {}:".format(epoch) if steps == -1 else " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Information Retrieval Evaluation on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model, *args, **kwargs)

        ##Log the scores
        for key in scores:
            logger.info(key + ": " + str(scores[key]))
            if self.wandb_names is not None:
                wandb.log({key: scores[key]})


        # Write results to disc
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(scores[name]['accuracy@k'][k])

                for k in self.precision_recall_at_k:
                    output_data.append(scores[name]['precision@k'][k])
                    output_data.append(scores[name]['recall@k'][k])

                for k in self.mrr_at_k:
                    output_data.append(scores[name]['mrr@k'][k])

                for k in self.ndcg_at_k:
                    output_data.append(scores[name]['ndcg@k'][k])

                for k in self.map_at_k:
                    output_data.append(scores[name]['map@k'][k])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        

        if self.main_score_function is None:
            print("Using max accuracy@k as main score - ", scores[name]['accuracy@k'][1])
            return max([scores[name]['accuracy@k'][1] for name in self.score_function_names])
        
        else:
            print("Using max accuracy@k as main score - ", scores[name]['accuracy@k'][1])
            return scores[self.main_score_function]['accuracy@k'][1]    

    def calculate_matching_accuracy(self, queries_result_list) -> float:
        matches = []
        for query_itr in range(len(self.queries)):
            query = self.queries[query_itr]
            relevant_docs = self.relevant_docs_dic[query]
            top_k_results = queries_result_list[query_itr]

            for doc in top_k_results:
                corpus_id = doc['corpus_id']
                if corpus_id in relevant_docs:
                    match = {"query": query, "match": corpus_id, "score": doc['score']}
                    matches.append(match)
                    break

        matching_accuracy = len(matches) / len(self.queries)

        # Save the matching results to a CSV file
        matching_df = pd.DataFrame(matches)
        matching_df.to_csv("matching_output.csv", index=False)

        return matching_accuracy



    def compute_metrices(self, model, corpus_model = None, corpus_embeddings: Tensor = None) -> Dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k), max(self.precision_recall_at_k), max(self.map_at_k))

        # Compute embedding for the queries
        query_embeddings = model.encode(self.queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size, convert_to_tensor=True,normalize_embeddings=True)

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        #Iterate over chunks of the corpus
        for corpus_start_idx in trange(0, len(self.corpus), self.corpus_chunk_size, desc='Corpus Chunks', disable=not self.show_progress_bar):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(self.corpus))

            #Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = corpus_model.encode(self.corpus[corpus_start_idx:corpus_end_idx], show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True)
            else:
                sub_corpus_embeddings = corpus_embeddings[corpus_start_idx:corpus_end_idx]

            #Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                #Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(pair_scores, min(max_k, len(pair_scores[0])), dim=1, largest=True, sorted=False)
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(pair_scores_top_k_idx[query_itr], pair_scores_top_k_values[query_itr]):
                        corpus_id = self.corpus_ids[corpus_start_idx+sub_corpus_id]
                        if len(queries_result_list[name][query_itr]) < max_k:
                            heapq.heappush(queries_result_list[name][query_itr], (score, corpus_id))  # heaqp tracks the quantity of the first element in the tuple
                        else:
                            heapq.heappushpop(queries_result_list[name][query_itr], (score, corpus_id))
        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {'corpus_id': corpus_id, 'score': score}
        logger.info("Queries: {}".format(len(self.queries)))
        logger.info("Corpus: {}\n".format(len(self.corpus)))

        #Compute scores
        scores = {name: self.compute_metrics(queries_result_list[name]) for name in self.score_functions}

        #Output
        for name in self.score_function_names:
            logger.info("Score-Function: {}".format(name))
            self.output_scores(scores[name])

        return scores


    def compute_metrics(self, queries_result_list: List[object],save_best_matches=True):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        best_match_dict={k_matches:{} for k_matches in self.accuracy_at_k}


        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]
            query_text= self.queries_raw[query_id]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break
            #Fill up best_match_dict
            for k_val in self.accuracy_at_k:
                ##Make a tuple with correct match or not (1/0) and the hit
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        best_match_dict[k_val][query_text]=(1,self.corpus_raw[hit['corpus_id']])
                        break
                    else:
                        best_match_dict[k_val][query_text]=(0,self.corpus_raw[hit['corpus_id']])
            

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])

        ##Save the best matches as a json
        if save_best_matches:
            # with open('best_matches.json', 'w') as fp:
            #     json.dump(best_match_dict, fp)
            
            ##Make a df with the matches. keys, value[0], value[1] as the three columns
            df = pd.DataFrame(list(best_match_dict[1].items()), columns=["Query", "Match"])
            df["Correct"] = df["Match"].apply(lambda x: x[1])
            df["Match"] = df["Match"].apply(lambda x: x[0])

            ###save the df as a csv 
            df.to_csv("current_best_matches.csv", index=False,encoding="utf-8")

 
        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}




class BinaryClassificationEvaluator_wandb(evaluation.BinaryClassificationEvaluator):


    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int], name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True, wandb_names=None):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.wandb_names = wandb_names

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "cossim_accuracy", "cossim_accuracy_threshold", "cossim_f1", "cossim_precision", "cossim_recall", "cossim_f1_threshold", "cossim_ap",
                            "manhattan_accuracy", "manhattan_accuracy_threshold", "manhattan_f1", "manhattan_precision", "manhattan_recall", "manhattan_f1_threshold", "manhattan_ap",
                            "euclidean_accuracy", "euclidean_accuracy_threshold", "euclidean_f1", "euclidean_precision", "euclidean_recall", "euclidean_f1_threshold", "euclidean_ap",
                            "dot_accuracy", "dot_accuracy_threshold", "dot_f1", "dot_precision", "dot_recall", "dot_f1_threshold", "dot_ap"]



    def compute_metrices(self, model):

        print("*********************EVALUATION STARTED******************************")

        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

        labels = np.asarray(self.labels)
        output_scores = {}
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True], ['manhattan', 'Manhattan-Distance', manhattan_distances, False], ['euclidean', 'Euclidean-Distance', euclidean_distances, False], ['dot', 'Dot-Product', dot_scores, True]]:
            # Note: newer versions of sbert have updated the spelling on manhatten to manhattan

            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                'accuracy': acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
            if self.wandb_names:
                wandb.log({
                    f"Classification Accuracy {name}": acc,
                    f"Classification Accuracy threshold {name}": acc_threshold,
                    f"Classification F1 {name}": f1,
                    f"Classification F1 threshold {name}": f1_threshold,
                    f"Classification Precision {name}": precision,
                    f"Classification Recall {name}": recall,
                    f"Classification Average precision {name}": ap
                })

        return output_scores