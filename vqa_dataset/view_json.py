import json

file_name = "answer_mapping.json"

with open(file_name) as f:
    data = json.load(f)

# Print the first 3 items with indentation
for item in data[:3]:  # Adjust slicing if it's a dictionary or list
    print(json.dumps(item, indent=4))
