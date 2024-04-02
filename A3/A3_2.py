import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np


# load train, dev, test data

class CustomDataset(Dataset):
    def __init__(self, words, tags, word2idx, tag2idx):
        self.words = words
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        tag = self.tags[idx]
        return self.word2idx[word], self.tag2idx[tag]


def read_ner_data(path_to_file):
    words = []
    tags = []
    with open(path_to_file, 'r', encoding='utf-8') as file:
        for line in file:
            splitted = line.split()
            if len(splitted) == 0:
                continue
            word = splitted[0]
            if word == '-DOCSTART-':
                continue
            entity = splitted[-1]
            words.append(word)
            tags.append(entity)
        return words, tags


TRAIN_PATH = 'data/train.txt'
DEV_PATH = 'data/dev.txt'
TEST_PATH = 'data/test.txt'

train_words, train_tags = read_ner_data(TRAIN_PATH)
dev_words, dev_tags = read_ner_data(DEV_PATH)
test_words, test_tags = read_ner_data(TEST_PATH)

# Convert all words to lowercase
train_words = [word.lower() for word in train_words]
dev_words = [word.lower() for word in dev_words]
test_words = [word.lower() for word in test_words]

# Build vocabularies for words and labels
word_vocab = set(train_words + dev_words + test_words)
label_vocab = set(train_tags + dev_tags + test_tags)

print('Word vocabulary size:', len(word_vocab))
print('Tag vocabulary size:', len(label_vocab))

# Define mappings from words and labels to indices
word2idx = {word: idx for idx, word in enumerate(word_vocab)}
label2idx = {label: idx for idx, label in enumerate(label_vocab)}



train_dataset = CustomDataset(train_words, train_tags, word2idx, label2idx)
dev_dataset = CustomDataset(dev_words, dev_tags, word2idx, label2idx)
test_dataset = CustomDataset(test_words, test_tags, word2idx, label2idx)

batch_size = 32  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)





# # Define a data loader that returns batches
# def collate_fn(batch):
#     sentences, labels = zip(*batch)
#     sentence_lengths = [len(sentence) for sentence in sentences]
#     max_length = max(sentence_lengths)
#     padded_sentences = []
#     for sentence in sentences:
#         padded_sentence = [word2idx[word] for word in sentence]
#         padded_sentence += [0] * (max_length - len(sentence))
#         padded_sentences.append(padded_sentence)
#     return torch.LongTensor(padded_sentences), torch.LongTensor(labels), torch.LongTensor(sentence_lengths)



# load the pretrained embedding data
def read_embedding_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    print('已读入')
    word2vec = {}
    # 遍历每一行内容
    for i, line in enumerate(lines):
        # 利用空格分割每一行，获取单词和对应的embedding向量
        parts = line.split()
        word = parts[0]
        embedding = np.array([float(x) for x in parts[1:]])
        word2vec[word] = embedding
    return word2vec



emb_size = 100  # Set the desired embedding dimension
embedding_file = 'glove.6B/glove.6B.100d.txt'  # Path to the pretrained embedding file

# Load the pretrained embeddings
word2vec = read_embedding_file(embedding_file)


# build model
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_size, hidden_dim, num_layers, pretrained_embeddings=None):
        super(BiLSTMClassifier, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


# train
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(labels).to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Set hyperparameters
hidden_dim = 128
num_layers = 2
vocab_size = len(word_vocab)
tag_size = len(label_vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the model
model = BiLSTMClassifier(vocab_size, tag_size, emb_size, hidden_dim, num_layers)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 2
best_accuracy = 0.0

for epoch in range(num_epochs):
    avg_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}")

    # Evaluation on validation set
    accuracy = evaluate(model, dev_loader, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%")

    # Save the best model based on validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

# Load the best model and evaluate on the test set
best_model = BiLSTMClassifier(vocab_size, tag_size, emb_size, hidden_dim, num_layers)
best_model.load_state_dict(torch.load("best_model.pth"))
best_model.to(device)

# accuracy = evaluate(best_model, test_loader, device)
# print(f"Test Accuracy: {accuracy:.2f}%")
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = torch.tensor(inputs).to(device)
        labels = torch.tensor(labels).to(device)

        outputs = best_model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Compute the F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print(f"Test F1 score: {f1:.4f}")