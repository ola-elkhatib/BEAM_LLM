import pickle
from tqdm import tqdm
import pandas as pd
rel_file="/storage/ola/BeamQA/entity_relation_prediction.pkl"

with open(rel_file, 'rb') as file:
    relation_entity_prediction = pickle.load(file)

data_path = '/storage/ola/BeamQA/Data/QA_data/WQSP/test_wqsp_original.txt'
test_data = pd.read_csv(data_path, sep='\t', names=['qa', 'ans', 'rel'])
datas = test_data.values
present_count=0
not_present_count=0
missing=0
for k in tqdm(range(len(datas))):
    data_sample = datas[k]
    h = data_sample[0].strip().split('[')
    topic_entity = h[1].split(']')[0]
    try:
        path = datas[k][2].split("|")[0]
    except:
        missing+=1
        continue
    breakpoint()
    if path in relation_entity_prediction[topic_entity]:
        present_count+=1
    else:
        not_present_count+=1

print("present: ", present_count)
print("not_present_count: ", not_present_count)
print