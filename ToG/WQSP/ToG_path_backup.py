from tqdm import tqdm
from utils import *
import random
from client import *
import logging
def data_generator(data,beam_data, entity2idx):
    df2 = pd.read_csv(beam_data,index_col=0,delimiter='\t',header=0,names=['qa','rel','scores','hopscores']) #paths generated using Path_generation module
    df2 = df2.values
    for i in range(len(data)):
        data_sample = data[i]
        h = data_sample[0].strip().split('[')
        head = h[1].split(']')[0]
        if head in entity2idx : head = entity2idx[head]
        else : head = -1
        if str(data_sample[1]) != 'nan' :
            ans = data_sample[1].split('|')
        else : ans = 'nan'
        scores = ast.literal_eval(df2[i][2])
        scores = [float(a) for a in scores] #path score
        relations = df2[i][1].split('|') # paths generated
        yield torch.tensor(head, dtype=torch.long), ans, relations , scores

def ToG_on_Graph_Path(cfg,entity2idx,entity2label,datas,nx_graph):
    total_correct=0
    pure_LLM_answers=0
    ToG_answers=0
    # for entity in entity2idx:
    #     if entity not in entity2label:
    #         print(entity)
    for k in tqdm(range(len(datas))):
        data_sample = datas[k]
        h = data_sample[0].strip().split('[')
        topic_entity = h[1].split(']')[0]
        question=h[0]
    
        if str(data_sample[1]) != 'nan' :
            ans = data_sample[1].split('|')
        else : ans = 'nan'


        ans_labels=[]
        for a in ans:
            label=id2entity_name_or_type(a)
            ans_labels.append(label)
            

        path = datas[k][2]

        logging.info('+ ----------------------------------------------------------------- +')
        logging.info(f'Question: {question}')
        logging.info(f'Topic Entity: {topic_entity}')
        logging.info(f'True Path: {path}')
        logging.info(f'True answer: {ans}, Labels: {ans_labels}')
        topic_entity=[topic_entity]
        cluster_chain_of_entities = []
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        for depth in range(1, cfg.depth+1):
            
            logging.info(f'\t At depth: {depth}')
            current_entity_relations_list = []
            i=0
            for entity_id in topic_entity:
                if entity_id!="[FINISH_ID]":
                    entity_name=id2entity_name_or_type(entity_id)
                    logging.info(f'\t\tTopic entity: {entity_name}')
                    #Relation Exploration
                    retrieve_relations_with_scores = relation_search_prune(nx_graph,entity_name, entity_id, pre_relations, pre_heads[i], question, cfg)  # best entity triplet, entitiy_id
                    logging.info(f'\t\tretrieve_relations_with_scores: {retrieve_relations_with_scores}')
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            for entity in current_entity_relations_list:
                logging.info(f'\t\t Entity: {entity}')
                breakpoint()
                if entity['head']:
                    entity_candidates_id = entity_search(nx_graph,entity['entity'], entity['relation'], True)
                else:
                    #Get all entities connected to head in the graph
                    entity_candidates_id = entity_search(nx_graph,entity['entity'], entity['relation'], False)
                
                if len(entity_candidates_id) >=20:
                    entity_candidates_id = random.sample(entity_candidates_id, cfg.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue
                entity_candidates=[id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id ]
                relation=entity['relation']
                logging.info(f'\t\t\tentity_candidates for {relation}: {entity_candidates}')
                scores, entity_candidates, entity_candidates_id = entity_score(entity2label,question, entity_candidates_id, entity['score'], entity['relation'], cfg)
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores,entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
            breakpoint()
            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, cfg)
                break
            #Getting the top entities across the different relations
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(entity2label,total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, cfg)
            logging.info(f'\t After entity pruning: {chain_of_entities}')
            breakpoint()
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, cfg)
                logging.info(f'\t\t Reasoning: {results}')
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    logging.info("ToG stopped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=cfg.dataset)
                    ToG_answers=extract_predicted_entity_count(results,ans_labels,ToG_answers)
                    logging.info(f"ToG_answers: {ToG_answers}")
                    logging.info(f"Hits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
                    logging.info(f"i={i}")
                    flag_printed = True
                    break
                
                else:
                    print("depth %d still not find the answer." % depth)
                    logging.info("depth %d still not find the answer." % depth)
                    topic_entity = entities_id
                    continue
            else:
                half_stop(question, cluster_chain_of_entities, cfg)
        

        if not flag_printed:
            results = generate_without_explored_paths(question, cfg)
            save_2_jsonl(question, results, [], file_name=cfg.dataset)
            pure_LLM_answers=extract_predicted_entity_count(results,ans_labels,pure_LLM_answers)
            logging.info(f"pure_LLM_answers: {pure_LLM_answers}")
            logging.info(f"Hits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
        logging.info('+ ----------------------------------------------------------------- +')

    accuracy = (ToG_answers+pure_LLM_answers) / len(datas)
    print('Hits@1 ',accuracy)
    print('Pure_LLM_answers count= ',pure_LLM_answers)
    print('ToG_answers count= ',ToG_answers)
    return accuracy