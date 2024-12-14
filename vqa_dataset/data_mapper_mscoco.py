import json

def create_combined_json(questions_file, annotations_file, output_file):
    try:
        # Load questions JSON
        with open(questions_file, 'r') as q_file:
            questions_data = json.load(q_file)
        
        # Load annotations JSON
        with open(annotations_file, 'r') as a_file:
            annotations_data = json.load(a_file)
        
        # Prepare a dictionary of annotations for quick lookup
        annotations_dict = {ann['question_id']: ann for ann in annotations_data['annotations']}
        
        # Combine the data
        combined_data = []
        total_len = len(questions_data['questions'])
        
        for idx, question in enumerate(questions_data['questions']):
            print(f'\r{idx}/{total_len}', end='')
            
            question_id = question['question_id']
            image_id = question['image_id']
            question_text = question['question']
            
            # Get corresponding annotation
            if question_id in annotations_dict:
                annotation = annotations_dict[question_id]
                combined_data.append(
                    {'questions': question_text} | annotation
                )
            else:
                print(f"NO QUESTION ID FOUND FOR {question_id=}")
        
        # Save the combined data to a new JSON file
        with open(output_file, 'w') as out_file:
            json.dump(combined_data, out_file, indent=4)
        
        print(f"Combined JSON saved to: {output_file}, {len(combined_data)=}")
    except Exception as e:
        print(f"Error: {e}")

# Filenames
questions_file = "v2_OpenEnded_mscoco_train2014_questions.json"
annotations_file = "v2_mscoco_train2014_annotations.json"
output_file = "combined_data_train.json"

# Run the function
create_combined_json(questions_file, annotations_file, output_file)
