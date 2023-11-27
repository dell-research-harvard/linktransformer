import pandas as pd
import pickle
import numpy as np
import linktransformer as lt
from linktransformer.cluster_fns import clusters_from_edges

###run as script 

if __name__ == "__main__":
    
    dataset_path=lt.DATA_DIR_PATH+"/es_mexican_products.xlsx"

    ###Open the dataframe
    df = pd.read_excel(dataset_path)

    df=df.copy()
    ##Drop missing descriptions
    df=df.dropna(subset=["description47","description48"])
    ##Make descriptions lowercase
    df=df.applymap(lambda x: x.lower() if isinstance(x,str) else x)
    ##Drop dropna rows
    df=df.dropna()
    left_id_name=["tariffcode47"]
    right_id_name=["tariffcode48"]

    ###Group by left id name and give a id_l as group id
    df["id_l"]=df.groupby(left_id_name).ngroup()
    ###Group by right id name and give a id_r as group id
    df["id_r"]=df.groupby(right_id_name).ngroup()

    df["id_l"]=df["id_l"].astype(str)
    df["id_r"]=df["id_r"].astype(str)

    ###Add a suffix _l and _r to the id_l and id_r columns
    df["id_l"]=df["id_l"]+"_l"
    df["id_r"]=df["id_r"]+"_r"


    left_id_rename="id_l"
    right_id_rename="id_r"

    ###Now 
    edge_list = list(zip(df[left_id_rename], df[right_id_rename]))
    cluster_assignment = clusters_from_edges(edge_list)
    ## Assign cluster ids to each node
    ## Now, make a mapping between left_id and cluster id - reverse the dict
    cluster_assignment = {k: v for v, l in cluster_assignment.items() for k in l}
    df["cluster_assignment"] = df[left_id_rename].map(cluster_assignment)

    ###Convert this to a 2 column df where we have text, cluster_assignment  - so converting it from wide to long format
    df = df.melt(id_vars=["cluster_assignment"], value_vars=["description47", "description48"], value_name="description", var_name="year")

    ####now, let's create a df where we deduplicate by cluster_assignment
    df_deduped = df.drop_duplicates(subset=["cluster_assignment"])

    ###to find the nearest cluster, lt.merge_knn with itself and k=8
    df_knn_deduped = lt.merge_knn(df_deduped,df_deduped, k=4, left_on=["description"], right_on=["description"],model="hiiamsid/sentence_similarity_spanish_es")

    ##Drop if id_l_x==id_l_y
    # df_knn_deduped=df_knn_deduped[df_knn_deduped["id_l_x"]!=df_knn_deduped["id_l_y"]]

    ###Now keep only cluster_assignment_x, cluster_assignment_y
    df_knn_deduped=df_knn_deduped[["cluster_assignment_x","cluster_assignment_y"]]

    df_knn_deduped.to_csv("df_deduped.csv")

    # ###Now, merge this with the original df on cluster assignment_x and cluster_assignment 
    df_knn_deduped=df_knn_deduped.merge(df,left_on=["cluster_assignment_x"],right_on=["cluster_assignment"])

    df_knn_deduped.to_csv("df_deduped.csv")
    df_knn_deduped=df_knn_deduped.drop(columns=["cluster_assignment"])
    ##Now merge with the original df on cluster_assignment_y and cluster_assignment
    df_knn_deduped=df_knn_deduped.merge(df,left_on=["cluster_assignment_y"],right_on=["cluster_assignment"])
    df_knn_deduped=df_knn_deduped.drop(columns=["cluster_assignment"])

    ##Drop exact matches
    df_knn_deduped=df_knn_deduped[df_knn_deduped["description_x"]!=df_knn_deduped["description_y"]]


    df_knn_deduped.to_csv("df_deduped.csv")


    print(df_knn_deduped)
