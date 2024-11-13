import re
def extract_failing_questions(f1_path, f2_path, f3_path):
    # Open the file and read its contents
    with open(f1_path, 'r') as file:
        file_content = file.read()

    # Define a pattern to capture text following "Question:"
    pattern = r"Question:\s*(.*)"

    # Find all matches in the file content
    questions = re.findall(pattern, file_content)
    questions=[question.lower() for question in questions]



    with open(f2_path, 'r') as f2:
        f2_lines = f2.readlines()


    with open(f3_path, 'w') as f3:
        for line in f2_lines:
            # Extract the question part before the first '['
            question_f2 = line.split(' [')[0].strip()
            if question_f2 not in questions:
                    f3.write(line+"\n")

# Example usage:
f1_path = 'logging_WQSP-full_2_embedding_space_bakcup.log'
f2_path = '/storage/ola/BeamQA/Data/QA_data/WQSP/test_wqsp_original.txt'
f3_path = 'missing_questions.txt'

extract_failing_questions(f1_path, f2_path, f3_path)
