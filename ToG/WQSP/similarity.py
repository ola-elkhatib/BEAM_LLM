import numpy as np
from loaders import  CheckpointLoader
import torch
from Model import Model
import torch.nn.functional as F
import pickle

kg_model_path = '../../Data/Graph_data/Freebase/'
kg_model_name = 'checkpoint_fb_full.pt'
nx_graph_path = '../../Data/Graph_data/Freebase/FB_full.gpickle'

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

def get_entity_embedding(entity_id, embedding_matrices):
    entity_embeddings = embedding_matrices[0]  # Entity embedding matrix
    return entity_embeddings[entity_id]

def get_relation_embedding(relation_id, embedding_matrices):
    relation_embeddings = embedding_matrices[1]  # Relation embedding matrix
    return relation_embeddings[relation_id]

def get_top_k_related_relations(entity_id, embedding_matrices, idx2rel, k=5):
    # Get the entity embedding vector
    #entity_embeddings = torch.tensor(embedding_matrices[0]) 
    entity_embeddings = torch.stack([torch.tensor(vec) for vec in embedding_matrices[0]])

    # Get the relation embeddings matrix
    #relation_embeddings = torch.tensor(embedding_matrices[1])  
    relation_embeddings = torch.stack([torch.tensor(vec) for vec in embedding_matrices[1]])
    entity_embedding = entity_embeddings[entity_id]

    # Compute dot product between the entity embedding and each relation embedding
    dot_products = torch.matmul(relation_embeddings, entity_embedding)

    # Get the top k relations with the highest dot product values
    top_k_indices = torch.topk(dot_products, k=k).indices
    top_k_relations = [(idx2rel[idx.item()], dot_products[idx].item()) for idx in top_k_indices]

    return top_k_relations

embedding_matrices , entity2idx, rel2idx , idx2rel = get_embeddings(kg_model_path,kg_model_name,256)
model = Model(embedding_matrices,0.2,True,True).to('cpu')
nx_graph = load_graph(nx_graph_path)
topic_entity=['m.0110g35j', 'm.01265cw2', 'm.01hjwf', 'm.01pj_5', 'm.026hxwx', 'm.02ntb8', 'm.02pjfgv', 'm.02q5hqz', 'm.02q674n', 'm.02qdrjx', 'm.03ck20l', 'm.03cn85q', 'm.03qhxn_', 'm.03ywfs', 'm.040_ypz', 'm.0479qny', 'm.04gbj8', 'm.04j08hn', 'm.04j0wxq', 'm.04j1hm9', 'm.04j2m_x', 'm.04rk7s', 'm.056p3k', 'm.05nd09x', 'm.066b_n', 'm.072txz', 'm.07h9gp', 'm.07zm5t', 'm.080kmxp', 'm.08pqpg', 'm.09t10b', 'm.0b5mw7', 'm.0b64qt1', 'm.0bh8xz2', 'm.0c02882', 'm.0cr_bqp', 'm.0crrvj7', 'm.0crv9zg', 'm.0crvchf', 'm.0crw06', 'm.0d042l', 'm.0d2l2_', 'm.0d9z_7', 'm.0f0sjn', 'm.0gxsyb5', 'm.0h3w444', 'm.0h944f1', 'm.0tl8nwg', 'm.0tl8ql6', 'm.0tl8srv', 'm.0tl8vtw']
tp='m.0110g35j' #label: Dragula

rel_graph = [i[2] for i in nx_graph.out_edges(tp, data='data')]
breakpoint()
s_id = entity2idx[tp]
tail_id= entity2idx["m.02822"]
    #rel_id = rel2idx[rel]
s = torch.Tensor([s_id]).long().to("cpu") 
t = torch.Tensor([tail_id]).long().to("cpu")            # subject indexes
    #p = torch.Tensor([rel_id]).long().to(device)          # relation indexes
#scores=model.score_relations_with_tails(s)
#scores=model.score_relations_with_tails_libkge(s)
#scores = torch.softmax(scores ,dim=1)
#scores=model.ComplEx_relation(s)
scores=model.ComplEx_relation_tail(s,t)
#sc, o = torch.topk(scores, 50, largest=True, dim=-1)  # index of highest-scoring objects
sc, o = torch.topk(scores, 50, largest=True, dim=0) 

#ans = [idx2rel[rel] for rel in o.tolist()[0]]
ans = [idx2rel[rel[0]] for rel in o.tolist()]
temp=[s[0] for s in sc.tolist()]
answr_score = dict(zip(ans,temp))

result= [k for k, v in sorted(answr_score.items(), key=lambda item: item[1], reverse=True)]
count_present=0
count_not_present=0
for rel in rel_graph: 
    if rel in result:
        print(rel, " Present in topk")
        count_present+=1
    else:
        print(rel, " Not Present in topk")
        count_not_present+=1
print("present= ", count_present)
print("not present= ", count_not_present)
breakpoint()

# tp_id=entity2idx[tp]
# breakpoint()
# idx2rel = {v: k for k, v in rel2idx.items()}
# top_k_relations=get_top_k_related_relations(tp_id,embedding_matrices, idx2rel )
# print(top_k_relations)