import argparse
import logging
import yaml
from ToG_path import *
from ToG_embedding_space import *
from Model import Model
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,  default="config.yaml", help="choose the config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    cfg = argparse.Namespace(dataset=config['dataset'], 
                                kg_type=config['kg_type'],
                                hops=config['hops'],
                                max_length=config['max_length'],
                                temperature_exploration=config['temperature_exploration'],
                                temperature_reasoning=config['temperature_reasoning'],
                                depth=config['depth'],
                                remove_unnecessary_rel=config['remove_unnecessary_rel'],
                                LLM_type=config['LLM_type'],
                                opeani_api_keys=config['opeani_api_keys'],
                                num_retain_entity=config['num_retain_entity'],
                                prune_tools=config['prune_tools'],
                                wandb_use=config['wandb_use'],
                                wandb_name=config['wandb_name'],
                                wandb_project=config['wandb_project'],
                                relation_search_tool=config['relation_search_tool'],
                                traversal_space=config['traversal_space'],
                                dropout=config['dropout'],
                                do_batchnorm=config['do_batchnorm'],
                                do_dropout=config['do_dropout'],
                                device=config['device'],
                                boost_rel_score_from_graph=config['boost_rel_score_from_graph'],
                                embedding_dimension=config['embedding_dimension'],
                                dataset_max=config['dataset_max'],
                                constraints_refuse=config['constraints_refuse'],
                                debug=config['debug'],
                                entity_width=config['entity_width'],
                                rel_width=config['rel_width']
                                )
    
    log_file_name=get_log_file_name(cfg)
    logging.basicConfig(filename=log_file_name, level=logging.INFO)


    #data_path = '../../Data/QA_data/WQSP/test_wqsp.txt'
    data_path = '/storage/ola/BeamQA/Data/QA_data/WQSP/test_wqsp_original.txt'
    #data_path='/storage/ola/BeamQA/ToG/WQSP/test_question.txt'
    #data_path = '/storage/ola/BeamQA/ToG/WQSP/failing_questions_formated_2.txt'
    nx_graph_path = '../../Data/Graph_data/Freebase/FB_'+ cfg.kg_type +'.gpickle'
    kg_model_path = '../../Data/Graph_data/Freebase/'
    kg_model_name = 'checkpoint_fb_' + cfg.kg_type + '.pt'
    #entity2label_path= '../../Data/Graph_data/Freebase/entity2label.txt'
    nx_graph = load_graph(nx_graph_path)
    #weave.init(cfg.wandb_project)
    
    embedding_matrices , entity2idx, rel2idx , idx2rel = get_embeddings(kg_model_path,kg_model_name,cfg.embedding_dimension)
 
    test_data = pd.read_csv(data_path, sep='\t', names=['qa', 'ans', 'rel'])
    datas = test_data.values

if cfg.traversal_space=='graph_path':
    accuracy=ToG_on_Graph_Path(cfg,entity2idx,datas,nx_graph)
elif cfg.traversal_space=='embedding_space':
    model = Model(embedding_matrices,cfg.dropout,cfg.do_batchnorm,cfg.do_dropout).to(cfg.device)
    accuracy=ToG_on_Embedding_Space(cfg,entity2idx,datas,nx_graph, rel2idx,model,cfg.device)
