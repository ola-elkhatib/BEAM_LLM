def extract_failing_questions(f1_path, f2_path, f3_path):
    # Read f1 and f2 into memory
    with open(f1_path, 'r') as f1:
        f1_lines = f1.readlines()
    
    with open(f2_path, 'r') as f2:
        f2_lines = f2.readlines()

    # Extract failing questions from f1
    failing_questions = []
    for line in f1_lines:
        if "question:" in line.lower():
            question = line.lower().split("question:")[1].strip()
            failing_questions.append(question)
    
    # Prepare to write to f3
    with open(f3_path, 'w') as f3:
        for line in f2_lines:
            # Extract the question part before the first '['
            question_f2 = line.split(' [')[0].strip()
            
            # Check if the question in f2 matches any failing question from f1
            if question_f2 in failing_questions:
                f3.write(line)

# Example usage:
f1_path = 'filtered_sorry.log'
f2_path = '/storage/ola/BeamQA/Data/QA_data/WQSP/test_wqsp_original.txt'
f3_path = 'unanswered_questions.txt'

extract_failing_questions(f1_path, f2_path, f3_path)
