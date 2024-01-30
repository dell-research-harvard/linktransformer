import os
import sys
from sentence_transformers.readers import InputExample
import logging

logger = logging.getLogger(__name__)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

    


def load_data_as_pairs(data, type):

    """
    This is for contrastive or other pairwise losses.
    """

    source_text_gt=data['left_text'].tolist()
    target_text_gt=data['right_text'].tolist()
    label=data['label'].tolist()
    label2int = {"same": 1, "different": 0, 1: 1, 0: 0}

    paired_data = []
    for i in range(len(source_text_gt)):
        label_id = label2int[label[i]]
        paired_data.append(InputExample(texts=[source_text_gt[i], target_text_gt[i]], label=float(label_id)))

    print(f'{len(paired_data)} {type} pairs')

    return paired_data


def load_data_as_individuals(cluster_dict, type):
    '''
    This is for SupCon Loss

    '''
  
    
    indv_data = []
    guid = 1
    clus_id=1
    for cluster_id in list(cluster_dict.keys()):
        

        for text in cluster_dict[cluster_id]:
            indv_data.append(InputExample(guid=guid, texts=[text], label=clus_id))
            guid += 1
        clus_id +=1

    print(f'{len(indv_data)} {type} examples')
    return indv_data

