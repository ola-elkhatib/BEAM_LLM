#from freebase_func import *
from prompt_list import *
import json
import re
import time
from openai import OpenAI
#from rank_bm25 import BM25Okapi
from loaders import  CheckpointLoader
#from sentence_transformers import util
import torch
#from loaders import  CheckpointLoader
#import networkx as nx
import logging
from get_missing_entities import *
#import pandas as pd
import pickle
import random
from transformers import AutoTokenizer, AutoModel
from beamQA import entity_search_embedding_space
import torch
from torch.nn.functional import cosine_similarity
import os
import unicodedata

#entity2label_path= '../../Data/Graph_data/Freebase/entity2label.txt'
entity2label_full_path="../../Data/Graph_data/Freebase/entity2labels_all.txt"
entity2label_path= '../../Data/Graph_data/Freebase/entity2label.txt'
relation_prune_file_path='relation_prune_cache.gpickle'
generate_answer_file_path="generate_answer_cache.gpickle"
reasoning_file_path="reasoning_cache.gpickle"
force_answer_file_path="force_answer_cache.gpickle"

def get_cached_reponses():
    
    if os.path.exists(relation_prune_file_path):
        with open(relation_prune_file_path, 'rb') as f:
            relation_prune_cache = pickle.load(f)
    else:
        relation_prune_cache={}
    
    if os.path.exists(generate_answer_file_path):
        with open(generate_answer_file_path, 'rb') as f:
            generate_answer_cache = pickle.load(f)
    else:
        generate_answer_cache={}

    if os.path.exists(reasoning_file_path):
        with open(reasoning_file_path, 'rb') as f:
            reasoning_cache = pickle.load(f)
    else:
        reasoning_cache={}

    if os.path.exists(force_answer_file_path):
        with open(force_answer_file_path, 'rb') as f:
            force_answer_cache = pickle.load(f)
    else:
        force_answer_cache={}
    return [relation_prune_cache, 0], [generate_answer_cache,0], [reasoning_cache,0], [force_answer_cache,0]

def save_responses(relation_prune_cache, generate_answer_cache, reasoning_cache, force_answer_cache) :
    with open(relation_prune_file_path, 'wb') as f1:
        pickle.dump(relation_prune_cache,f1)
    with open(generate_answer_file_path, 'wb') as f2:
        pickle.dump(generate_answer_cache,f2)
    with open(reasoning_file_path, 'wb') as f3:
        pickle.dump(reasoning_cache,f3)
    with open(force_answer_file_path, 'wb') as f4:
        pickle.dump(force_answer_cache,f4)

def prepare_entity_labels(file_path):
    data_dict = {}
    # Open the file in read mode
    with open(file_path, 'r') as file:
        for line in file:
            # Strip any leading/trailing whitespace and split by tab
            if len(line.strip().split('\t'))>1:
                key, value = line.strip().split('\t')
            # Add the key-value pair to the dictionary
            modified_key = key.replace('/', '', 1).replace('/', '.', 1)
            data_dict[modified_key] = value
    return data_dict

entity2label_1=prepare_entity_labels(entity2label_path)   
entity2label_full=prepare_entity_labels(entity2label_full_path)   

def id2entity_name_or_type(entity_id):
    if entity_id in entity2label_1:
        return entity2label_1[entity_id]
    elif entity_id in entity2label_full:
        return entity2label_full[entity_id]
    else:
        entity_id = re.sub(r'^m\.', r'/m/', entity_id)
        label=get_missing_entities_label(entity_id)
        if label:
            return label
        return "UnName_Entity"

    

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

# def compute_bm25_similarity(query, corpus, width=3):
#     """
#     Computes the BM25 similarity between a question and a list of relations,
#     and returns the topn relations with the highest similarity along with their scores.

#     Args:
#     - question (str): Input question.
#     - relations_list (list): List of relations.
#     - width (int): Number of top relations to return.

#     Returns:
#     - list, list: topn relations with the highest similarity and their respective scores.
#     """

#     tokenized_corpus = [doc.split(" ") for doc in corpus]
#     bm25 = BM25Okapi(tokenized_corpus)
#     tokenized_query = query.split(" ")

#     doc_scores = bm25.get_scores(tokenized_query)
    
#     relations = bm25.get_top_n(tokenized_query, corpus, n=width)
#     doc_scores = sorted(doc_scores, reverse=True)[:width]

#     return relations, doc_scores


def clean_relations(string, entity_id, head_relations,tail_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        #assert relation in head_relations, "The chosen relation {} by LLM is not from the list provided".format(relation)

        if head_relations and relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        elif tail_relations and relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
        else:
            print("Relation not chose from list: ",relation)
            logging.info(f'\t\tRelation not in relation list: {relation}')

                    
    if not relations:
        #return False, "No relations found"
        return False, relations
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    for i in range(len(topn_relations)):
        if topn_relations[i] in head_relations:
            relations.append({"entity": entity_id, "relation": topn_relations[i], "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": topn_relations[i], "score": topn_scores[i], "head": False})
    return True, relations

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-4"):
    # if "llama" not in engine.lower():
    #     openai.api_key = "EMPTY"
    #     openai.api_base = "http://localhost:8000/v1"  # your local llama server port
    #     engine = openai.Model.list()["data"][0]["id"]
    # else:
    #     openai.api_key = opeani_api_keys

    client = OpenAI(api_key=opeani_api_keys)
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    print("start openai")
    f=0
    while(f == 0):
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=engine,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0)

            result = chat_completion.choices[0].message.content

            f = 1
        except:
            print("openai error, retry")
            time.sleep(2)
    print("end openai")
    return  result

def construct_relation_prune_prompt(question, entity_name, total_relations, cfg):
    return extract_relation_prompt % (cfg.width, cfg.width) + question + '\nTopic Entity of Q2: ' + entity_name + '\nFor Q2, retrieve from the following relations:: '+ '; '.join(total_relations) + "\nA2: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):

    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '

def relation_search_prune(nx_graph,entity_label, entity_id, pre_relations, pre_head, question, cfg): 
    ##############Relation Search#########################
    head_relations=[]
    for i in nx_graph.out_edges(entity_id, data='data'):
        head_relations.append(i[2])

    #Get the relations from the incoming edges to head entity
    tail_relations=[]
    for i in nx_graph.in_edges(entity_id, data='data'):
         tail_relations.append(i[2])

    if cfg.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if len(pre_relations) != 0 and pre_head !=-1:
        tail_relations = [rel for rel in tail_relations if not pre_head or rel not in pre_relations]
        head_relations = [rel for rel in head_relations if pre_head or rel not in pre_relations]
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    ################Relation Prune##########################
    if cfg.relation_search_tool == "llm":
        prompt = construct_relation_prune_prompt(question, entity_label, total_relations, cfg)
        if relation_prune_cache[(question, entity_label)]:
            result=relation_prune_cache[(question, entity_label)]
        else:
            result = run_llm(prompt, cfg.temperature_exploration, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
            relation_prune_cache[(question, entity_label)]=result
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations, tail_relations)

    # elif cfg.relation_search_tool == "bm25":
    #     topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, cfg.width)
    #     flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations,tail_relations) 

    elif cfg.relation_search_tool == "bert":
        topn_relations, topn_scores = compute_bert_relevance_scores(question, total_relations)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations,tail_relations) 
    # else:
    #     model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
    #     topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, cfg.width)
    #     flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] # format error or too small max_length
    
    
def entity_search(nx_graph,entity_id, relation, head=True): 
    if head:
        entities_id=[]
        for i in nx_graph.out_edges(entity_id, data='data'):
            if i[2]==relation:
                entities_id.append(i[1])
    else:
        entities_id=[]
        for i in nx_graph.in_edges(entity_id, data='data'):
            if i[2]==relation:
                entities_id.append(i[0])
    return entities_id


def entity_score(question, entity_candidates_id, score, relation, cfg):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    # if all_unknown_entity(entity_candidates):
    #     return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates,entity_candidates_id, []
    entity_candidates_filtered,entity_candidates_id_filtered, entity_candidates_ids_deleted = del_unknown_entity(entity_candidates,entity_candidates_id, score)
   
    
    if len(entity_candidates_filtered) == 1:
        return [score], entity_candidates_filtered, entity_candidates_id_filtered, entity_candidates_ids_deleted
    if len(entity_candidates_filtered) == 0:
        return [0.0], entity_candidates_filtered, entity_candidates_id_filtered, entity_candidates_ids_deleted
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates_filtered, entity_candidates_id_filtered))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)


    if cfg.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)
        result = run_llm(prompt, cfg.temperature_exploration, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
        return [float(x) * score for x in clean_scores(result, entity_candidates)],entity_candidates, entity_candidates_id, entity_candidates_ids_deleted

    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id, entity_candidates_ids_deleted

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)

def check_if_literal(ans, ans_labels):
    for i in range(len(ans)):
        if not ans[i].startswith(("m.", "g.")):
            ans_labels[i]=ans[i]
    return ans_labels

def entity_filter_unknowns(entity_candidates_ids, scores):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_ids]
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates, [],scores,[],[]
    entity_candidates_filtered=[]
    entity_candidates_deleted=[]
    score_filtered=[]
    score_deleted=[]
    entity_candidates_ids_filtered=[]
    for i in range(len(entity_candidates)):
        if entity_candidates[i] == "UnName_Entity":
            entity_candidates_deleted.append(entity_candidates_ids[i])
            score_deleted.append(scores[i])

        else:
            entity_candidates_filtered.append(entity_candidates[i])
            entity_candidates_ids_filtered.append(entity_candidates_ids[i])
            score_filtered.append(scores[i])
    return entity_candidates_filtered,entity_candidates_ids_filtered,score_filtered, entity_candidates_deleted, score_deleted

def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)

def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, entity_candidates_ids_deleted, score_deleted, total_deleted_relations, total_deleted_entities, total_score_deleted, total_deleted_topic_entities):           
    candidates_relation = [entity['relation']] * len(entity_candidates)
    deleted_relations=[entity['relation']] * len(entity_candidates_ids_deleted)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    total_score_deleted.extend(score_deleted)
    total_deleted_relations.extend(deleted_relations)
    total_deleted_entities.extend(entity_candidates_ids_deleted)
    total_deleted_topic_entities.extend([entity['entity']] * len(deleted_relations))
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, total_deleted_relations, total_deleted_entities, total_deleted_topic_entities


def generate_answer(reasoning_cache_list, question, cluster_chain_of_entities, cfg): 
    cluster_chain_of_entities_tuple=tuple(map(tuple, cluster_chain_of_entities))
    if (question, cluster_chain_of_entities_tuple) in reasoning_cache_list[0]:
        result=reasoning_cache_list[0][(question, cluster_chain_of_entities_tuple)]
    else:
        prompt = force_answer + question + '\n'
        chain_prompt = '\n'.join([', '.join(f"'{i}'" for i in item) for item in cluster_chain_of_entities])
        prompt += "\nKnowledge Triplets: " + chain_prompt + 'A6: '
        result = run_llm(prompt, cfg.temperature_reasoning, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
        reasoning_cache_list[0][(question,cluster_chain_of_entities_tuple)]=result
        reasoning_cache_list[1]+=1

    return result


def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities}
    with open("ToG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def entity_prune(total_candidate_ids, total_relations, total_candidates, total_topic_entities, total_head, total_scores, cfg):
    zipped = list(zip(total_candidate_ids, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:cfg.width], sorted_relations[:cfg.width], sorted_candidates[:cfg.width], sorted_topic_entities[:cfg.width], sorted_head[:cfg.width], sorted_scores[:cfg.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], []
    
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list)) 
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]
    return True, cluster_chain_of_entities, entities_id, relations


def reasoning(cache_list, question, cluster_chain_of_entities, cfg, LLM_calls):
    LLM_calls+=1
    #cluster_chain_of_entities_tuple=tuple(map(tuple, cluster_chain_of_entities))
    cluster_chain_of_entities_tuple = tuple(sorted(tuple(t) for t in cluster_chain_of_entities))

    if (question, cluster_chain_of_entities_tuple) in cache_list[0]:
        response=cache_list[0][(question, cluster_chain_of_entities_tuple)]
    else:
        prompt = prompt_evaluate + "\n\nQ5: " +question
        #chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
        chain_prompt = '\n'.join([', '.join(f"'{i}'" for i in item) for item in cluster_chain_of_entities])
        prompt += "\nKnowledge Triplets: " + chain_prompt + 'A5: '
        response = run_llm(prompt, cfg.temperature_reasoning, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
        cache_list[0][(question,cluster_chain_of_entities_tuple)]=response
        cache_list[1]+=1
        
    result = extract_answer(response)
    if if_true(result):
        return True, response, LLM_calls
    else:
        return False, response, LLM_calls
    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    
def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False

def half_stop(force_answer_cache_list, question, cluster_chain_of_entities, cfg):
    print("No new knowledge added during search depth %d or max depth is reached, stop searching." % cfg.depth)
    logging.info(f'"No new knowledge added during search depth {cfg.depth} or max depth is reached, stop searching. ')
    answer = generate_answer(force_answer_cache_list, question, cluster_chain_of_entities, cfg)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=cfg.dataset)
    return answer


def generate_without_explored_paths(generate_answer_cache_list, question, cfg, LLM_calls):
    LLM_calls+=1
    if (question) in generate_answer_cache_list[0]:
        return generate_answer_cache_list[0][(question) ], LLM_calls
    else:
        prompt = generate_directly + "\n\nQ: " + question + "\nA:"
        response = run_llm(prompt, cfg.temperature_reasoning, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
        generate_answer_cache_list[0][(question) ]=response
        generate_answer_cache_list[1]+=1
        
    return response, LLM_calls



def process_text_file(text_file):
   # what movies are about [ginger rogers]	Top Hat|Kitty Foyle|The Barkleys of Broadway	has_tags_inv
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        try:
            question_data={}
            data_line = data_line.strip()
            if data_line == '':
                continue
            data_line = data_line.strip().split('\t')
            question_data['question']=data_line[0]
            question = data_line[0].split('[')
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            ans = data_line[1].split('|')
            path = data_line[2].split('|')

        
            question_data['topic_entity']=[head,]
            question_data['answer']=ans
            question_data['path']=path
            data_array.append(question_data)
        except:
            breakpoint()
    return data_array[1:]


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

def entity_search_embed(head, rel,pscore,model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = 5 ,retscore = False):

    entities_and_score=entity_search_embedding_space(head, rel, model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk, retscore = True)

    # this happens when we dont have an embedding for the head or the relation
    # In this case, we choose to traverse the graph instead of the embedding space
    if entities_and_score==[(None,None)]:
        entity_ids=entity_search(nx_graph,head, rel, head=True)
        if len(entity_ids)>5:
            entity_ids=random.sample(entity_ids, topk)
        if len(entity_ids)==0:
            return []
        entities_and_score=[(id,1) for id in entity_ids]

    entities_and_score=[(entity, score*pscore) for (entity, score) in entities_and_score]
    
    return entities_and_score

def clean_results(string):
    matches = re.findall(r'\{([^}]+)\}', string)
    predicted_entities="NULL"
    if matches:
        predicted_entities=re.split(',|;|\|| and',matches[-1])
        predicted_entities=[entity.strip() for entity in predicted_entities if entity]
    return predicted_entities
    if "{" in string:
        start = string.find("{") + 1
        end = string.find("}")
        content = string[start:end]
        return content
    else:
        return "NULL"
def check_string(string):
    return "{" in string

def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False

def extract_predicted_entity_count(results,ans,total_correct,total_failing,total_not_answered, attempted=False):
    logging.info(f"\t\t\t results: {results}")
    answered=False
    correctly_answered=False
    if check_string(results):
        response = clean_results(results)
        if response=="NULL":
            print("response is NULL NO match")
            logging.info(f"\t\t\t Question NOT ANSWERED")
            response = results
            if attempted:
                total_not_answered+=1
        else:
            answered=True
            predicted_entities=[entity.strip("\"' .") for entity in response]
            for a in predicted_entities:
                if exact_match(a, ans):
                    total_correct += 1
                    logging.info(f"\t\t\t CORRECT ANSWER: {a}")
                    correctly_answered=True
                    break
            if not correctly_answered:
                logging.info(f"\t\t\t WRONG ANSWER: {predicted_entities}")
                total_failing+=1
    else:
        logging.info(f"\t\t\t Check String Failed")
        if attempted:
            total_not_answered+=1
    #     response = results
    #     if cfg.constraints_refuse and check_string(response):
    #         return total_correct,total_failing, answered
    #     if exact_match(response, ans):
    #         total_correct += 1
    #         answered=True
    #     else:
    #         total_failing+=1
    return total_correct,total_failing,total_not_answered, answered
# def extract_predicted_entity_count(results,ans,total_correct):
#     matches = re.findall(r'\{([^}]+)\}', results)
#     answered=False
#     if matches:
#         predicted_entities=re.split(',|;|\|| and',matches[-1])
#         predicted_entities=[entity.strip("\"' .") for entity in predicted_entities]
#         for predicted_entity in predicted_entities:
#             if predicted_entity in ans:
#                 total_correct += 1
#                 answered=True
#                 return total_correct, answered

    print("Question not answered correctly by LLM")
    logging.info(f'\t\tQuestion not answered correctly by LLM')
    logging.info(f'\t\tLLM response: {results}')
    logging.info(f'\t\tanswers: {ans}')
    return total_correct, answered



def get_log_file_name(cfg):
    return f'logging_{cfg.dataset}-{cfg.kg_type}_{cfg.hops}_{cfg.traversal_space}.log'


def relation_search_prune_embed(relation_prune_cache_list, beam_data,entity_id, entity_name,relation_list, question, cfg,prev_rel,hop, LLM_calls):
    
    flag=True
    #Get the top 50 paths from the Path generation model
    relation_list_path=beam_data[1].split('|')
    #Get the scores of the top50 paths
    score_list_path=[float(score) for score in beam_data[2][1:-1].split(',')]
    if prev_rel is None:
        relation_list=[rel.split()[0] for rel in relation_list_path]
        score_list=score_list_path
    else:
        relation_list=[]
        score_list=[]
        for i in  range(len(relation_list_path)):
            rel=relation_list_path[i]
            path=rel.split()
            if len(path)>hop and rel.split()[0]==prev_rel:
                relation_list.append(rel.split()[hop])
                score_list.append(score_list_path[i])
    zipped = zip(relation_list, score_list)

    # Dictionary to store the highest score for each relation
    relation_score_dict = {}
    # Iterate over the zipped relations and scores
    for rel, score in zipped:
        # If the relation is already in the dictionary, update if the score is higher
        if rel in relation_score_dict:
            relation_score_dict[rel] = max(relation_score_dict[rel], score)
        else:
            # Otherwise, add the relation with its score
            relation_score_dict[rel] = score
    # Extract the relations and their highest scores as lists
    head_relations = list(relation_score_dict.keys())
    highest_scores = list(relation_score_dict.values())

    ################Relation Prune##########################

    if len(head_relations)>3:
        if cfg.relation_search_tool == "llm":
            LLM_calls+=1
            if (question, entity_name, tuple(head_relations)) in relation_prune_cache_list[0]:
                result=relation_prune_cache_list[0][(question, entity_name, tuple(head_relations))]
            else:
                prompt = construct_relation_prune_prompt(question, entity_name, head_relations, cfg)
                result = run_llm(prompt, cfg.temperature_exploration, cfg.max_length, cfg.opeani_api_keys, cfg.LLM_type)
                relation_prune_cache_list[0][(question, entity_name, tuple(head_relations))]=result
                relation_prune_cache_list[1]+=1
               
            flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations, None)



            #Combine the scores with scores from the Path generation module
            
            for e in retrieve_relations_with_scores:
                #e['score']=relation_score_dict[e['relation']]
                e['score']=relation_score_dict[e['relation']]
                #*e['score']
    
        # elif cfg.relation_search_tool == "bm25":
        #     topn_relations, topn_scores = compute_bm25_similarity(question, head_relations, cfg.width)
        #     flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations,None) 

        else:
            topn_relations, topn_scores = compute_bert_relevance_scores(question, head_relations, cfg.width)
            flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations,None) 
        if flag:
            return retrieve_relations_with_scores, LLM_calls
        else:
            return [], LLM_calls # format error or too small max_length
    else:
        relations=[]
        for rel in head_relations:
            relations.append({"entity": entity_id, "relation": rel, "score": score, "head": True})
        return relations, LLM_calls

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')    

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])",'', w)
    w = re.sub(r'(\[.*\])+', "", w)
    w = re.sub(r"[^a-zA-Z0-9_?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w =w.replace('\\','')
    return w