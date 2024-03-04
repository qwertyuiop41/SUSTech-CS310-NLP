import json

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
            id = json_data['id']

            processed_sentence = self._tokenize(sentence)
            processed_choices = [self._tokenize(choice) for choice in choices]

            processed_data.append({
                'sentence': processed_sentence,
                'label': label
            })
        return processed_data

    def _tokenize(self, text):
        # # Tokenize text by treating each single Chinese character as a token
        # tokens = re.findall(r'[\u4e00-\u9fff]', text)
        # return ''.join(tokens)
        #Improved
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        digit_pattern = re.compile(r'\d+')
        english_word_pattern = re.compile(r'[a-zA-Z]+')
        # 匹配除了中英文数字空格之外的特殊字符
        punctuation_pattern = re.compile(r'[^\u4e00-\u9fff\da-zA-Z\s]')
        tokens= re.findall(r'[\u4e00-\u9fff]|\d+|[a-zA-Z]+|[^\u4e00-\u9fff\da-zA-Z\s]', text)
        return ''.join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# Example usage
train_dataset = CustomDataset('train.jsonl')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_iterator = iter(train_dataloader)


count = 0
for item in train_dataloader:
    print(item)
    count += 1
    if count > 7:
        break
#
# def basic_tokenizer(text):
#     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
#     tokens = []
#     for char in text:
#         if chinese_pattern.match(char):
#             tokens.append(char)
#     return tokens
#
#
# def yield_basic_tokens(data_iter):
#     for text, _ in data_iter:
#         yield basic_tokenizer(text)
#
#
# count = 0
# for tokens in yield_basic_tokens(train_iterator): # Use a new iterator
#     print(tokens)
#     count += 1
#     if count > 7:
#         break
#
#
#
# def improved_tokenizer(text):
#     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
#     digit_pattern = re.compile(r'\d+')
#     english_word_pattern = re.compile(r'[a-zA-Z]+')
#     #匹配除了中英文数字空格之外的特殊字符
#     punctuation_pattern = re.compile(r'[^\u4e00-\u9fff\da-zA-Z\s]')
#     tokens = []
#     for token in re.findall(r'[\u4e00-\u9fff]|\d+|[a-zA-Z]+|[^\u4e00-\u9fff\da-zA-Z\s]', text):
#             tokens.append(token)
#     return tokens
#
#
#
# def yield_improved_tokens(data_iter):
#     for text, _ in data_iter:
#         yield improved_tokenizer(text)
#
#
# count = 0
# for tokens in yield_basic_tokens(train_iterator): # Use a new iterator
#     print(tokens)
#     count += 1
#     if count > 7:
#         break
#



# import json
# import re
#
# def jsonl_iterator(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             json_data = json.loads(line)
#             yield json_data
#
# jsonl_file = 'train.jsonl'
# train_iterator = jsonl_iterator(jsonl_file)
#
# count = 0
# for item in train_iterator:
#     print(item)
#     count += 1
#     if count > 7:
#         break
#
#
# #################
# def basic_tokenizer(text):
#     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
#     tokens = []
#     for char in text:
#         if chinese_pattern.match(char):
#             tokens.append(char)
#     return tokens
#
#
# def yield_basic_tokens(data_iter):
#     for text, _ in data_iter:
#         yield basic_tokenizer(text)
#
#
# count = 0
# for tokens in yield_basic_tokens(train_iterator): # Use a new iterator
#     print(tokens)
#     count += 1
#     if count > 7:
#         break
#
#
#
# def improved_tokenizer(text):
#     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
#     digit_pattern = re.compile(r'\d+')
#     english_word_pattern = re.compile(r'[a-zA-Z]+')
#     #匹配除了中英文数字空格之外的特殊字符
#     punctuation_pattern = re.compile(r'[^\u4e00-\u9fff\da-zA-Z\s]')
#     tokens = []
#     for token in re.findall(r'[\u4e00-\u9fff]|\d+|[a-zA-Z]+|[^\u4e00-\u9fff\da-zA-Z\s]', text):
#             tokens.append(token)
#     return tokens
#
#
#
# def yield_improved_tokens(data_iter):
#     for text, _ in data_iter:
#         yield improved_tokenizer(text)
#
#
# count = 0
# for tokens in yield_basic_tokens(train_iterator): # Use a new iterator
#     print(tokens)
#     count += 1
#     if count > 7:
#         break
