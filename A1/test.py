import json
import re
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = self._process_data(file_path)

    def _process_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        processed_data = []
        for line in lines:
            json_data = json.loads(line)
            sentence = json_data['sentence']
            choices = json_data['choices']
            label = json_data['label'][0]
            example_id = json_data['id']

            processed_sentence = self._tokenize(sentence)
            processed_choices = [self._tokenize(choice) for choice in choices]

            processed_data.append({
                'sentence': processed_sentence,
                'label': label
            })

        return processed_data

    def _tokenize(self, text):
        # Tokenize text by treating each single Chinese character as a token
        tokens = re.findall(r'[\u4e00-\u9fff]', text)
        return ''.join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# Example usage
train_dataset = CustomDataset('train.jsonl')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

count = 0
for item in train_dataloader:
    print(item)
    count += 1
    if count > 7:
        break