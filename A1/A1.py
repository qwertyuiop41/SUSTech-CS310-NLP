
import json
import re
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

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

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
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
embedding_dim = 256
hidden_dim = 512
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



def train(model, dataloader, optimizer, criterion, epoch: int):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (labels, token_ids, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(token_ids, offsets)
        try:
            loss = criterion(output, labels)
        except Exception:
            print('Error in loss calculation')
            print('output: ', output.size())
            print('labels: ', labels.size())
            # print('token_ids: ', token_ids)
            # print('offsets: ', offsets)
            raise
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (output.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            output = model(text, offsets)
            loss = criterion(output, label)
            total_acc += (output.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count



# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 8  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# First, obtain some output and labels
model.eval()
with torch.no_grad():
    for i, (labels, token_ids, offsets) in enumerate(train_dataloader):
        output = model(token_ids, offsets)
        # print(f"batch {i} output: {output}")
        if i == 0:
            break

loss = criterion(output, labels)
print('loss:', loss)

criterion2 = torch.nn.CrossEntropyLoss(reduction='none')
loss2 = criterion2(output, labels)
print('loss non-reduced:', loss2)
print('mean of loss non-reduced:', torch.mean(loss2))

# Manually calculate the loss
probs = torch.exp(output[0,:]) / torch.exp(output[0,:]).sum()
loss3 = -torch.log(probs[labels[0]])
print('loss manually computed:', loss3)





# Prepare train, valid, and test data
train_dataset = myDataset('train.jsonl')
test_dataset = myDataset('test.jsonl')
# train_dataset = to_map_style_dataset(train_iter)
# test_dataset = to_map_style_dataset(test_iter)

num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
### Main Training Loop

# Run the training loop
total_accu = None
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    print('$$$$$$$$$$$$$$$$$')
    train(model, train_dataloader, optimizer, criterion, epoch)
    accu_val = evaluate(model, valid_dataloader, criterion)

    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val

    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)


# Save the model
torch.save(model.state_dict(), "text_classification_model.pth")
accu_test = evaluate(model, valid_dataloader, criterion)
print("test accuracy {:8.3f}".format(accu_test))