with open('failing_questions_formated.txt', 'r') as f1:
    f1_lines = set(line.strip() for line in f1)

# Read the lines from F2
with open('/storage/ola/BeamQA/WQSP/failing_questions_formated.txt', 'r') as f2:
    f2_lines = set(line.strip() for line in f2)

# Count the number of lines in F1 that are also in F2
matching_lines = f1_lines.intersection(f2_lines)
print(f"Number of lines in F1 that exist in F2: {len(matching_lines)}")