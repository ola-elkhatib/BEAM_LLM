import re
from utils import *
# # Open the input file in read mode and the output file in write mode
# with open("../../Data/Graph_data/Freebase/english_labels.txt", "r") as infile, open("../../Data/Graph_data/Freebase/entity2labels_all.txt", "w") as outfile:
#     for line in infile:
#         # Use regex to capture entity ID and label
#         match = re.search(r"<http://rdf\.freebase\.com/ns/(m\.[^>]+)>\s+<http://www\.w3\.org/2000/01/rdf-schema#label>\s+\"([^\"]+)\"@en", line)
        
#         # If there’s a match, extract the entity ID and label
#         if match:
#             entity_id = match.group(1)
#             entity_label = match.group(2)
#             entity_id = re.sub(r'^m\.', r'/m/', entity_id)
#             # Write the result to the output file
#             outfile.write(f"{entity_id}\t{entity_label}\n")
entity2label_path="../../Data/Graph_data/Freebase/entity2labels_all.txt"
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


# nx_graph_path = '../../Data/Graph_data/Freebase/FB_full.gpickle'
# nx_graph = load_graph(nx_graph_path)
# ans=['m.02zzm_', 'm.0340r0', 'm.039rqy']
# for i in nx_graph.out_edges('m.05kkh', data='data'):
#     if i[2]=='government.governmental_jurisdiction.governing_officials':
#         if i[1] in ans:
#             print(i)
#         for j in nx_graph.out_edges(i[1], data='data'):
#             if j[2]=='government.government_position_held.office_holder':
#                 if j[1] in ans:
#                     print(i)
#                     print(j)
#                     print()


entity2label2=prepare_entity_labels(entity2label_path)   
breakpoint()