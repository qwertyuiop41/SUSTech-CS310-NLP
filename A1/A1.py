import json

def jsonl_iterator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            yield json_data

# Example usage
jsonl_file = 'train.jsonl'
train_iterator = jsonl_iterator(jsonl_file)

# Iterate over the data
for item in train_iterator:
    # Process each item
    sentence = item['sentence']
    choices = item['choices']
    label = item['label']
    example_id = item['id']
    # Perform further operations on the data
    # ...