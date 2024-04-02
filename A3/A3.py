from pprint import pprint
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


input_file = 'lunyu_20chapters.txt'

from utils import CorpusReader
corpus = CorpusReader(inputFileName=input_file, min_count=1)

word2id: dict = {}
id2word: dict = {}

word2id.update({'[PAD]': 0})
word2id.update({k: v+1 for k, v in corpus.word2id.items()})
id2word = {v: k for k, v in word2id.items()}

print(word2id['子'])
print(word2id['曰'])
print(word2id['。'])

lines = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        lines.append(list(line.strip()))



class RNNLM(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size ):
        # super(RNNLM, self).__init__()
        # self.embedding = nn.Embedding(kwargs['vocab_size'], kwargs['emb_size'])
        # self.rnn = nn.RNN(kwargs['emb_size'], kwargs['hidden_size'], batch_first=True)
        # self.fc = nn.Linear(kwargs['hidden_size'], kwargs['vocab_size'])

        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn=nn.RNN(emb_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, seq, seq_lens):
            embedded = self.embedding(seq)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)


            rnn_output, _ = self.rnn(packed)

            padded, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
            logits = self.fc(padded)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs


#######################################

def compute_perplexity(logits, targets):
    loss_fn = nn.NLLLoss(ignore_index=0, reduction='none')
    with torch.no_grad():
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        perplexity = torch.exp(loss.mean())
    return perplexity


# randomly initialized embedding + report perplexity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_size=50
vocab_size=len(word2id)
hidden_size=256
embedding_lunyu = nn.Embedding(len(word2id), emb_size)
model = RNNLM(vocab_size=vocab_size,emb_size=emb_size,hidden_size=hidden_size).to(device)


# 定义损失函数和优化器
criterion = nn.NLLLoss(ignore_index=0, reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.1)


def train(model,embedding):
    # 训练过程
    model.train()
    num_epochs = 1  # 迭代次数
    for epoch in range(num_epochs):

        optimizer.zero_grad()

        seq_ids = [torch.tensor([word2id.get(w, 0) for w in line], dtype=torch.long).to(device) for line in lines]
        seq_lens = torch.tensor([len(line) for line in seq_ids])
        seq_ids_padded = nn.utils.rnn.pad_sequence(seq_ids, batch_first=True).to(device)

        # seq_embs = embedding(seq_ids_padded)
        # seq_embs_packed = nn.utils.rnn.pack_padded_sequence(seq_embs, seq_lens, batch_first=True, enforce_sorted=False)
        #
        # out_packed,_= model(seq_embs_packed)
        # out_unpacked,_= nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        targets_padded = seq_ids_padded.clone()
        for i in range(len(targets_padded)):
            targets_padded[i, :-1] = targets_padded[i, 1:].clone()
            targets_padded[i, -1] = word2id.get('[PAD]', 0)

        log_probs = model(seq_ids_padded, seq_lens)
        loss = criterion(log_probs.view(-1, log_probs.shape[-1]), targets_padded.view(-1))
        loss.mean().backward()
        optimizer.step()
        perplexity = torch.exp(loss.mean())


        print(f"Epoch [{epoch+1}/{num_epochs}], loss: {loss.mean()}, perplexity:{perplexity}")




train(model,embedding_lunyu)
torch.save(model, "model.pth")

# generate sentences
def generate_sentence(model, start_tokens, end_token, max_length=20):
    model.eval()
    with torch.no_grad():
        start_ids = torch.tensor([word2id.get(w, 0) for w in start_tokens], dtype=torch.long).unsqueeze(0).to(device)
        current_ids=start_ids
        print('current ids: ')
        print(current_ids)
        generated_sentence = start_tokens.copy()

        for _ in range(max_length):
            log_probs=model(current_ids,[len(current_ids[0])])
            last_word_log_probs = log_probs[:, -1, :]
            predicted_id = torch.multinomial(torch.exp(last_word_log_probs.squeeze()), 1).item()
            print(predicted_id)
            predicted_word = id2word.get(predicted_id, "")
            generated_sentence.append(predicted_word)
            if predicted_word == end_token:
                break
            current_ids = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return "".join(generated_sentence)

start_tokens = ["子", "曰"]  # 开始标记组成的列表
end_token = "。"  # 结束标记的值
max_length = 20  # 生成句子的最大长度
sentence = generate_sentence(model, start_tokens, end_token, max_length)
print("Generated Sentence:", sentence)



######