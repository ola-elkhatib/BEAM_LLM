from tqdm import tqdm
from utils import *
from client import *
import logging
import numpy as np
import os
import networkx as nx

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def ToG_on_Embedding_Space(cfg,entity2idx,datas,nx_graph,rel2idx,model,device):
#Check if this is needed?
    #model.eval()
    
    relation_prune_cache_list, generate_answer_cache_list, reasoning_cache_list, force_answer_cache_list=get_cached_reponses()
    idx2entity = {v:k for k,v in entity2idx.items()}
    pure_LLM_answers=0
    ToG_answers=0
    failing_answers=0
    not_answered=0
    missing_information=0
    answer_unknown=0
    relation_list=rel2idx.keys()
    file_beam_data = 'wqsp_predictions_wscores_top50.txt'
    df2 = pd.read_csv(file_beam_data,index_col=0,delimiter='\t',header=0,names=['qa','rel','scores','hopscores']) #paths generated using Path_generation module
    df2 = df2.values
    
    for k in tqdm(range(len(datas))):

        LLM_calls=0
        data_sample = datas[k]
        h = data_sample[0].strip().split('[')
        topic_entity = h[1].split(']')[0]
        question=preprocess_sentence(h[0].strip())
        beam_data=df2[np.where(df2 == question)[0][0]]
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
        prev_rel=None
        topic_entity_relation=list(zip([topic_entity], [prev_rel]))
        
        for depth in range(1, cfg.depth+1):
            logging.info(f'\t At depth: {depth}')
            current_entity_relations_list = []

            for entity_id,prev_rel  in topic_entity_relation:
                entity_name=id2entity_name_or_type(entity_id)
                logging.info(f'\t\tTopic entity: {entity_id}')
                #Relation Search and Prune
                retrieve_relations_with_scores, LLM_calls = relation_search_prune_embed(relation_prune_cache_list, beam_data, entity_id,entity_name, relation_list, question, cfg, prev_rel,depth-1,LLM_calls)  # best entity triplet, entitiy_id
                logging.info(f'\t\tRelation scoring by LLM: {retrieve_relations_with_scores[:3]}')
                current_entity_relations_list.extend(retrieve_relations_with_scores)

            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            total_deleted_relations=[]
            total_deleted_candidate_ids=[]
            total_deleted_topic_entities=[]
            total_score_deleted=[]
            for entity in current_entity_relations_list:
                print(entity)
                logging.info(f'\t\tRelation Path of : {entity}')
                #Entity Search in embedding space
                entities_and_scores=entity_search_embed(entity['entity'], entity["relation"],entity['score'],model,rel2idx,entity2idx,idx2entity,nx_graph,device,topk = 5, retscore = True)
                logging.info(f'\t\t\tFrom embedding Space: ')
                logging.info(f'\t\t\t\tEntity_candidates: {entities_and_scores}')
                entity_candidates_id=[entity for (entity,_) in entities_and_scores]
                entity_candidates_scores=[score for (_,score) in entities_and_scores]
            
                #Remove Unnamed entities from list
                entity_candidates_filtered,entity_candidates_ids_filtered,score_filtered, entity_candidates_ids_deleted, score_deleted = entity_filter_unknowns(entity_candidates_id,entity_candidates_scores)
                logging.info(f'\t\t\t"Removing unNamed entities from list ')
                logging.info(f'\t\t\t\t"Entity Candidates: {entity_candidates_ids_filtered} and Scores: {score_filtered}')
                logging.info(f'\t\t\t\t"Deleted Candidates: {entity_candidates_ids_deleted} and Scores: {score_deleted}')

                total_candidates, total_scores, total_relations, total_entities_id, \
                total_topic_entities, total_head, total_deleted_relations, \
                total_deleted_candidate_ids, total_deleted_topic_entities = update_history(
                    entity_candidates_filtered, 
                    entity, 
                    score_filtered, 
                    entity_candidates_ids_filtered, 
                    total_candidates, 
                    total_scores, 
                    total_relations, 
                    total_entities_id, 
                    total_topic_entities, 
                    total_head, 
                    entity_candidates_ids_deleted, 
                    score_deleted,
                    total_deleted_relations, 
                    total_deleted_candidate_ids, 
                    total_score_deleted, 
                    total_deleted_topic_entities
                )                                                                                                                                                                 
            
            
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
                logging.info(f"\t\t\t Question  not answered through graph.")
                break
        
            flag, chain_of_entities, entities_id, pre_relations = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, cfg)
            
            logging.info(f'\t\tAfter entity pruning:\n \t\t\t\t\t{chain_of_entities}')
            cluster_chain_of_entities.extend(chain_of_entities)
            
            if flag and depth<cfg.depth:
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
                    break
                
                else:
                    logging.info("depth %d still not find the answer." % depth)
                    if depth<cfg.depth:
                        total_deleted_head=[True*len(total_deleted_topic_entities)]
                        total_deleted_candidates=["UnName_Entity"] * len(total_deleted_topic_entities)
                        total_targets=total_deleted_candidates+total_candidates
                        total_target_ids=total_deleted_candidate_ids+total_entities_id
                        total_target_relations=total_deleted_relations+total_relations
                        total_target_topic_entities=total_deleted_topic_entities+total_topic_entities
                        total_target_head=total_deleted_head+total_head
                        total_target_scores=total_score_deleted+total_scores
                        
                        flag, chain_of_entities, entities_id, pre_relations = entity_prune(total_target_ids, total_target_relations, total_targets, total_target_topic_entities, total_target_head, total_target_scores, cfg)
                        
                        logging.info(f"\t\tEntity Pruning after adding the unNanme and rescoring:")
                        logging.info(f"\t\t{chain_of_entities}")
                        
                        topic_entity_relation = list(zip(entities_id, pre_relations))
                        #deleted_unknowns=list(zip (total_deleted_candidates,total_deleted_relations))
                        #topic_entity_relation.extend(deleted_unknowns)
                        
                        cluster_chain_of_entities.extend(chain_of_entities)
                        logging.info(f"\t\tThe new cluster of entities list is: {cluster_chain_of_entities}")
                    #Option is to do entity pruning or scoring here
                    continue
            else:
                logging.info(f"\t\t\t Force to answer: {question}")
                logging.info(f"\t\t\t cluster_chain_of_entities: {cluster_chain_of_entities}")
                results=half_stop(force_answer_cache_list, question, cluster_chain_of_entities, cfg)
                ToG_answers,failing_answers,not_answered, answered=extract_predicted_entity_count(results,ans_labels,ToG_answers,failing_answers,not_answered)
                if answered:
                    if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                        print("a question is missed in count")
                    logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers}  Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
                    logging.info(f"\t\tHits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
                logging.info(f"\t\t\t Question  not answered through graph.")
                break


        if not answered:
            logging.info(f'\t\t\t ONLY LLM: {question}')
            results, LLM_calls = generate_without_explored_paths(generate_answer_cache_list,question, cfg, LLM_calls)
            save_2_jsonl(question, results, [], file_name=cfg.dataset)
            pure_LLM_answers,failing_answers, not_answered, answered=extract_predicted_entity_count(results,ans_labels,pure_LLM_answers,failing_answers, not_answered, True)
            if k+1!=ToG_answers+failing_answers+not_answered+ pure_LLM_answers+missing_information+answer_unknown:
                        print("a question is missed in count")
            if not answered:
                logging.info(f'Failing Question even by LLM: {question}')
            
            logging.info(f"\t\t\t Total questions: {k+1} pure_LLM_answers: {pure_LLM_answers} ToG_answers: {ToG_answers} Failing_answers: {failing_answers}  Not answered: {not_answered} Missing_information: {missing_information} Answer_unknown: {answer_unknown}")
            print("Failing_answers: ",failing_answers)
            print("pure_LLM_answers: ",pure_LLM_answers)
            logging.info(f"Hits@1: {(ToG_answers+pure_LLM_answers) / (k+1)}")
        
        #Every 10 questions save the dump in case it was new
        if k%10==0:
            if relation_prune_cache_list[1]>10 or  generate_answer_cache_list[1]>10 or reasoning_cache_list[1]>10:
                logging.info(f"Dumping cache files: relation_prune_cache_list:{relation_prune_cache_list[1]}, generate_answer_cache_list: {generate_answer_cache_list[1]}, reasoning_cache_list: {reasoning_cache_list[1]}, force_answer_list: {force_answer_cache_list[1]}")
                save_responses(relation_prune_cache_list[0], generate_answer_cache_list[0], reasoning_cache_list[0], force_answer_cache_list[0])
                relation_prune_cache_list[1]=0
                generate_answer_cache_list[1]=0
                reasoning_cache_list[1]=0
                force_answer_cache_list[1]=0
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
    save_responses(relation_prune_cache_list[0], generate_answer_cache_list[0], reasoning_cache_list[0], force_answer_cache_list[0])


    return accuracy