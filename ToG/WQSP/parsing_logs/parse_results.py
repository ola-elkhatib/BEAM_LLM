import re

f1_path = 'logging_WQSP-full_2_embedding_space_bakcup_copy.log'
f2_path = '/storage/ola/BeamQA/Data/QA_data/WQSP/test_wqsp_original.txt'
f3_path = 'missing_quesitons.txt'
with open(f1_path, 'r') as f1:
    f1_lines = f1.readlines()

    
log_data = ''.join(f1_lines)
# Regular expression pattern to capture text between the two delimiters
pattern = r"INFO:root:\+ ----------------------------------------------------------------- \+\n(.*?)INFO:root:\+ ----------------------------------------------------------------- \+"

# Find all matches
matches = re.findall(pattern, log_data, re.DOTALL)

# Filter out matches containing "Correct Answer"
filtered_matches_correct =0
filtered_matches_missing_information=0
failing_questons=0
remaining_questons=0
sorry_questions=0
missing=0
all_questions=[]
# Prepare to write to f3

for i, match in enumerate(matches, start=1):
    if "CORRECT ANSWER:" in match:
        filtered_matches_missing_information+=1
    elif "Question FAILED" in match:
        failing_questons+=1
    elif "Missing information about question in test set for question" in match:
        filtered_matches_missing_information+=1
    elif "All answer labels are unNamed" in match:
        missing+=1
    else: 
        remaining_questons+=1
        pattern_question = r"Question:\s*(.*)"
        questions = re.findall(pattern_question, match)

        all_questions.append(questions[0])

with open(f2_path, 'r') as f2:
        f2_lines = f2.readlines()
# Prepare to write to f3
with open(f3_path, 'w') as f3:
    for line in f2_lines:
            # Extract the question part before the first '['
            question_f2 = line.split(' [')[0].strip()
            
            # Check if the question in f2 matches any failing question from f1
            if question_f2 in all_questions:
                f3.write(line)

f3.write(match+"\n")
print(i)
print("filtered_matches_correct: ",filtered_matches_correct)
print("filtered_matches_missing_information: ", filtered_matches_missing_information)
print("failing_questons: ", failing_questons)
print("sorry_questions:", sorry_questions)
print("remaining_questons", remaining_questons)
total=filtered_matches_correct+filtered_matches_missing_information+failing_questons+sorry_questions+remaining_questons
print("total: ", total)