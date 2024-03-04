
import json
import re
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator

class myDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        processed_data = []
        for line in lines:
            json_data = json.loads(line)
            processed_data.append(json_data)
        self.data = processed_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def basic_tokenizer(sentence):
    tokens = re.findall(r'[\u4e00-\u9fff]', sentence)
    return tokens

def improved_tokenizer(sentence):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    digit_pattern = re.compile(r'\d+')
    english_pattern = re.compile(r'[a-zA-Z]+')
    # 匹配除了中英文数字空格之外的特殊字符
    punctuation_pattern = re.compile(r'[^\u4e00-\u9fff\da-zA-Z\s]')
    tokens = re.findall(r'[\u4e00-\u9fff]|\d+|[a-zA-Z]+|[^\u4e00-\u9fff\da-zA-Z\s]', sentence)
    return tokens

# Example usage
train_dataset = myDataset('train.jsonl')

train_iterator = iter(train_dataset)




def yield_tokens(data_iter):
    for item in data_iter:
        yield improved_tokenizer(item['sentence'])

count = 0
for tokens in yield_tokens(train_iterator): # Use a new iterator
    print(tokens)
    count += 1
    if count > 7:
        break



vocab = build_vocab_from_iterator(yield_tokens(train_iterator), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Print the vocabulary size
print("Vocabulary size:", len(vocab))

# Print a few random tokens and their corresponding indices
for token in ['保', '说', '，', '小']:
    print(f"Token: {token}, Index: {vocab[token]}")


text_pipeline = lambda x: vocab(improved_tokenizer(x))
label_pipeline = lambda x: int(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, token_ids_list, offsets = [], [], [0]
    for item in batch:
        label_list.append(label_pipeline(item['label'][0]))
        token_ids = torch.tensor(text_pipeline('sentence'), dtype=torch.int64)
        token_ids_list.append(token_ids)
        offsets.append(token_ids.size(0))

    labels = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    token_ids = torch.cat(token_ids_list)

    return labels.to(device), token_ids.to(device), offsets.to(device)



train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False,collate_fn=collate_batch)
count = 0
# Test the dataloader
for i, item in enumerate(train_dataloader):
    print(i)
    print(item)
    count += 1
    if count > 7:
        break



# Test the dataloader
for i, (labels, token_ids, offsets) in enumerate(train_dataloader):
    print(f"batch {i} label: {labels}")
    print(f"batch {i} text: {token_ids}")
    print(f"batch {i} offsets: {offsets}")
    if i == 0:
        break

# What does offsets mean?
print('Number of tokens: ', token_ids.size(0))
print('Number of examples in one batch: ', labels.size(0))
print('Example 1: ', token_ids[offsets[0]:offsets[1]])
print('Example 8: ', token_ids[offsets[7]:])


#######################################



class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(Model, self).__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        #指定两个隐藏层，每个隐藏层由nn.Linear和nn.ReLU激活函数组成。最后一层是线性层，输出num_classes个类别
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, token_ids, offsets):
        embedded = self.embedding(token_ids, offsets)
        output = self.fc(embedded)
        return output


# Example usage
vocab_size = len(vocab)
embedding_dim = 64
hidden_dim = 256
num_classes = 2 #0,1

model = Model(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)

model.eval()
with torch.no_grad():
    for i, (labels, token_ids, offsets) in enumerate(train_dataloader):
        output = model(token_ids, offsets)
        # print(f"batch {i} output: {output}")
        if i == 0:
            break

# Examine the output
print('output size:', output.size())
print('output:', output)


########################################################

