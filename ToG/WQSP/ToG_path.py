from tqdm import tqdm
from utils_path import *
from client import *
import logging
import numpy as np
import os
import networkx as nx

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def ToG_on_Graph_Path(cfg,entity2idx,datas,nx_graph):

    tog_relation_prune_cache_list, generate_answer_cache_list, reasoning_cache_list, force_answer_cache_list, entity_scoring_cache_list=get_cached_reponses()
    idx2entity = {v:k for k,v in entity2idx.items()}
    pure_LLM_answers=0
    ToG_answers=0
    pre_relations = [], 
    failing_answers=0
    not_answered=0
    missing_information=0
    answer_unknown=0
    
    for k in tqdm(range(len(datas))):

        LLM_calls=0
        data_sample = datas[k]
        h = data_sample[0].strip().split('[')
        topic_entity = h[1].split(']')[0]
        topic_entity=[topic_entity]
        question=preprocess_sentence(h[0].strip())
        pre_heads= [-1] * len(topic_entity)
        logging.info('+ ----------------------------------------------------------------- +')
        logging.info(f'Question: {question}')
        logging.info(f'Topic Entity: {topic_entity}')

        if str(data_sample[1]) != 'nan' :
            ans = str(data_sample[1]).split('|')
        else : 
            ans = 'nan'
            logging.info(f'Missing information about question in test set for question: {question}')
            missing_information+=1
            if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                print("a question is missed in count")
            continue
        
        ans_labels=[id2entity_name_or_type(a) for a in ans]
        if all_unknown_entity(ans_labels):
            logging.info(f'All answer labels are unNamed: {ans} {ans_labels}')
            ans_labels=check_if_literal(ans, ans_labels)
            if all_unknown_entity(ans_labels):
                answer_unknown+=1
                if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                    print("a question is missed in count")
                continue

        path = datas[k][2]
        logging.info(f'True Path: {path}')
        logging.info(f'True answer: {ans},  Labels: {ans_labels}')

        cluster_chain_of_entities = []
        flag_printed = False
        for depth in range(1, cfg.depth+1):
            logging.info(f'\t At depth: {depth}')
            current_entity_relations_list = []
            i=0
            for entity_id  in topic_entity:
                if entity_id!="[FINISH_ID]":
                    entity_name=id2entity_name_or_type(entity_id)
                    logging.info(f'\t\tTopic entity: {entity_id}')
                    #Relation Search and Prune
                    retrieve_relations_with_scores = relation_search_prune(tog_relation_prune_cache_list,nx_graph,entity_name, entity_id, pre_relations, pre_heads[i], question, cfg)  # best entity triplet, entitiy_id
                        
                    logging.info(f'\t\tRelation scoring by LLM: {retrieve_relations_with_scores}')
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1

            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            for entity in current_entity_relations_list:
                print(entity)
                logging.info(f'\t\tRelation Path of : {entity}')
                #Entity Search in embedding space
                if entity['head']:
                    entity_candidates_id = entity_search(nx_graph,entity['entity'], entity['relation'], True)
                else:
                    #Get all entities connected to head in the graph
                    entity_candidates_id = entity_search(nx_graph,entity['entity'], entity['relation'], False)
                
                if len(entity_candidates_id) >=20:
                    entity_candidates_id = random.sample(entity_candidates_id, cfg.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue
            
                logging.info(f'\t\t\t\tEntity_candidates: {entity_candidates_id}')
                entity_candidates=[id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id ]
                relation=entity['relation']
                logging.info(f'\t\t\tentity_candidates for {relation}: {entity_candidates}')
                scores, entity_candidates, entity_candidates_id = entity_score(entity_scoring_cache_list,question, entity_candidates_id, entity['score'], entity['relation'], cfg)
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores,entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
                                                                                                                                                          
            logging.info(f'\t\t"Total Entity Candidates: {total_candidates} and Scores: {total_scores}')
            if len(total_candidates) ==0:
                logging.info(f'\t\tHalf Stop: No new candidates added')
                logging.info(f"\t\t\t cluster_chain_of_entities: {cluster_chain_of_entities}")
                results=half_stop(force_answer_cache_list, question, cluster_chain_of_entities, cfg)
                ToG_answers,failing_answers,not_answered, answered=extract_predicted_entity_count(results,ans_labels,ToG_answers,failing_answers,not_answered)
                if answered:
                    if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                        print("a question is missed in count")
                    logging.info(f"\t\t\t Force to answer: {question}")
                    logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers} Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
                    logging.info(f"\t\tHits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
                break
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, cfg)
    
            logging.info(f'\t\tAfter entity pruning:\n \t\t\t\t\t{chain_of_entities}')
            cluster_chain_of_entities.extend(chain_of_entities)
            
            if flag:
                logging.info(f'\t\t Cluster chain: {cluster_chain_of_entities}')
                stop, results, LLM_calls = reasoning(reasoning_cache_list, question, cluster_chain_of_entities, cfg, LLM_calls)
                LLM_calls+=1
                logging.info(f'\t\t Reasoning: {results}')

                if stop:
                    logging.info("ToG stopped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=cfg.dataset)
                    logging.info("ans_labels %s." % ans_labels)
                    ToG_answers,failing_answers,not_answered, answered=extract_predicted_entity_count(results,ans_labels,ToG_answers,failing_answers,not_answered)
                    if answered:
                        if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                            print("a question is missed in count")
                       
                    logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers} Not_answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
                    print(f"ToG_answers: {ToG_answers}")
                    logging.info(f"\t\tHits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
                    flag_printed = True
                    break
                
                else:
                    logging.info("depth %d still not find the answer." % depth)
                    topic_entity = entities_id
                    #Option is to do entity pruning or scoring here
                    continue
            else:
                logging.info(f"\t\t\t cluster_chain_of_entities: {cluster_chain_of_entities}")
                results=half_stop(force_answer_cache_list, question, cluster_chain_of_entities, cfg)
                ToG_answers,failing_answers,not_answered, answered=extract_predicted_entity_count(results,ans_labels,ToG_answers,failing_answers,not_answered)
                if answered:
                    if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                        print("a question is missed in count")
                    logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers}  Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
                    logging.info(f"\t\tHits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")  
                break


        if not flag_printed:
            results, LLM_calls = generate_without_explored_paths(generate_answer_cache_list,question, cfg, LLM_calls)
            
            save_2_jsonl(question, results, [], file_name=cfg.dataset)
            pure_LLM_answers,failing_answers, not_answered, answered=extract_predicted_entity_count(results,ans_labels,pure_LLM_answers,failing_answers, not_answered, True)
            if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                print("a question is missed in count")
            logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers}  Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
            logging.info(f"Hits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
        logging.info('+ ----------------------------------------------------------------- +')

        #Every 10 questions save the dump in case it was new
        if k%10==0:
            if tog_relation_prune_cache_list[1]>10 or  generate_answer_cache_list[1]>10 or reasoning_cache_list[1]>10 or entity_scoring_cache_list[1]>10:
                logging.info(f"Dumping cache files: relation_prune_cache_list:{tog_relation_prune_cache_list[1]}, generate_answer_cache_list: {generate_answer_cache_list[1]}, reasoning_cache_list: {reasoning_cache_list[1]}, force_answer_list: {force_answer_cache_list[1]}, entity_scoring_cache_list: {entity_scoring_cache_list[1]}")
                save_responses(tog_relation_prune_cache_list[0], generate_answer_cache_list[0], reasoning_cache_list[0], force_answer_cache_list[0], entity_scoring_cache_list[0])
                tog_relation_prune_cache_list[1]=0
                generate_answer_cache_list[1]=0
                reasoning_cache_list[1]=0
                force_answer_cache_list[1]=0
                entity_scoring_cache_list[1]=0
        logging.info('+ ----------------------------------------------------------------- +')


    accuracy = (ToG_answers+pure_LLM_answers) / len(datas)
    print('Hits@1 ',accuracy)
    print('Total Correct ',ToG_answers+pure_LLM_answers)
    print('Pure_LLM_answers count= ',pure_LLM_answers)
    print('ToG_answers count= ',ToG_answers)
    print('Missing information in test count= ',missing_information)
    print('Answer label is UnNamed count= ',answer_unknown)
    logging.info(f"\t\t\t Total questions: {len(datas)} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers}  Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
    logging.info(f"\t\tHits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
    save_responses(tog_relation_prune_cache_list[0], generate_answer_cache_list[0], reasoning_cache_list[0], force_answer_cache_list[0], entity_scoring_cache_list[0])


    return accuracy