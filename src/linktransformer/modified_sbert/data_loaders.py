import os
import sys
from sentence_transformers.readers import InputExample
import logging

logger = logging.getLogger(__name__)

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

    


def load_data_as_individuals(cluster_dict, type,already_clustered=False):
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

