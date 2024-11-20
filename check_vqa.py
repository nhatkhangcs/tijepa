from collections import defaultdict
from datasets import load_dataset

ds = load_dataset("lmms-lab/VQAv2")

val_set = ds['validation']

# Count answer frequencies
answer_freq = defaultdict(int)

# Loop through training set
for i, item in enumerate(ds['validation']):
    print(f"{i}/{len(val_set)}", end="\r")
        
    # Get the most common answer for each question
    answer = item['multiple_choice_answer']
    answer_freq[answer] += 1

# Sort by frequency
sorted_answers = sorted(answer_freq.items(), key=lambda x: x[1], reverse=True)

# Get top 3129 answers
top_answers = set(ans for ans, freq in sorted_answers[:3129])
print(f"Number of top answers: {len(top_answers)}")