
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from loaders import  CheckpointLoader
import torch
from Model import Model
import torch.nn.functional as F
import pickle


def load_graph(path):
    #nx_graph = nx.read_gpickle(path)
    with open(path, 'rb') as f:
        nx_graph = pickle.load(f)
    return nx_graph
def get_embeddings(path,model_name,embedding_dim):
    '''

    :param path:
    :param model_name:
    :param embedding_dim:
    :return:
    '''
    ckp = CheckpointLoader(path)
    entity2idx, rel2idx, embedding_matrix, embedding_matrix_rel = ckp.load_libkge_checkpoint(model_name,embedding_dim)
    embedding_matrix_rel.append(torch.zeros(embedding_matrix_rel[0].shape[0]))  ### this is a padding embedding
    print('Ent ', len(embedding_matrix), len(embedding_matrix_rel), embedding_matrix[0].shape,
          embedding_matrix_rel[0].shape)

    embedding_matrices = [embedding_matrix, embedding_matrix_rel]
    idx2rel = {k:v for v,k in rel2idx.items()}
    idx2rel[len(idx2rel)] = 'None'
    rel2idx['None'] = len(rel2idx)

    return embedding_matrices , entity2idx, rel2idx , idx2rel

def compute_bert_relevance_scores(question, relations):
    # Step 1: Load a pre-trained transformer model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Step 3: Tokenize the question and relations
    question_tokens = tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    relation_tokens = [tokenizer(rel, return_tensors='pt', truncation=True, padding=True) for rel in relations]

    # Step 4: Encode the question and relations using the transformer model
    with torch.no_grad():
        question_embedding = model(**question_tokens).last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
        relation_embeddings = [model(**rel).last_hidden_state[:, 0, :] for rel in relation_tokens]

    # Step 5: Compute relevance scores using cosine similarity
    scores = [cosine_similarity(question_embedding, rel_emb)[0].item() for rel_emb in relation_embeddings]
    
    # Step 6: Combine relations with their scores and sort them by score in descending order
    scored_relations = sorted(zip(relations, scores), key=lambda x: x[1], reverse=True)

    rel_score=[]
    for pair in scored_relations:
        rel_score.append({"relation":pair[0], "score":pair[1]})

    return rel_score

kg_model_path = '../../Data/Graph_data/Freebase/'
kg_model_name = 'checkpoint_fb_full.pt'
nx_graph_path = '../../Data/Graph_data/Freebase/FB_full.gpickle'

embedding_matrices , entity2idx, rel2idx , idx2rel = get_embeddings(kg_model_path,kg_model_name,256)

question="what movies has carmen electra been in"
relations=[rel for rel in rel2idx.keys()]

#rel=compute_bert_relevance_scores(question,relations)
tp='m.0110g35j' #label: Dragula
nx_graph = load_graph(nx_graph_path)
rel_graph = [i[2] for i in nx_graph.out_edges(tp, data='data')]
breakpoint()
entity_dict={}
for node in nx_graph.nodes():
    for i in nx_graph.out_edges(node, data='data'):
        category=(".").join(i[2].split('.')[:-1])
        if node in entity_dict:
            entity_dict[node][category]+=1
        else:
            entity_dict[node]={}
            entity_dict[node][category]=1
breakpoint()
rel_dict={}
for rel in relations:
    category=(".").join(rel.split('.')[:-1])
    sub_rel=rel.split('.')[-1]
    if rel not in rel_dict:
        rel_dict[category]=[sub_rel]
    else:
        rel_dict[category].append(sub_rel)
breakpoint()
# count_present=0
# count_not_present=0
# for rel in rel_graph: 
#     if rel in top_relations:
#         print(rel, " Present in topk")
#         count_present+=1
#     else:
#         print(rel, " Not Present in topk")
#         count_not_present+=1
# print("present= ", count_present)
# print("not present= ", count_not_present)
# breakpoint()


