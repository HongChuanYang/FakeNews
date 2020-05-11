from torchnlp.nn import Attention

from conf import AppConf
from model.bceloss_penalty import BCEPLoss
from model.simple_cnn import SimpleCNN
import torch
import torch.nn as nn


class WordSentenceEncoding(nn.Module):
    def __init__(self, vocab_size, class_size, embedding_dim, embedding_matrix, device):
        super(WordSentenceEncoding, self).__init__()
        self.device = device
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            weight = torch.from_numpy(embedding_matrix).type(torch.FloatTensor).to(device)
            self.emb = nn.Embedding.from_pretrained(weight)
        self.emb.weight.requires_grad = True
        self.hidden_dim = embedding_dim
        self.layer_dim = 2
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim // 2,
                              bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, 10),
            nn.Tanh(),
            nn.Linear(10, class_size),
        )

        self.fc_f = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(AppConf.sentenceEncoding_r + 1, class_size)
        )

        self.d_a = AppConf.sentenceEncoding_r
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.loss = BCEPLoss(self.device)
        self.attention = Attention(self.hidden_dim, attention_type='general')
        self.w_s1 = nn.Parameter(torch.randn(self.d_a, self.hidden_dim))
        self.w_s2 = nn.Parameter(torch.randn(AppConf.sentenceEncoding_r, self.d_a))

    def init_hidden(self, batch_size):
        return (nn.Parameter(torch.zeros(self.layer_dim, batch_size, self.hidden_dim // 2).to(self.device)),
                nn.Parameter(torch.zeros(self.layer_dim, batch_size, self.hidden_dim // 2).to(self.device)))

    def forward(self, x, l):
        emb = self.emb(x)
        self.hidden = self.init_hidden(x.size(0))
        outputs, self.hidden = self.bilstm(emb, self.hidden)
        # word embedding
        features_last = [
            self.attention(h[l - 1].unsqueeze(0).unsqueeze(0), h[0:l, :].unsqueeze(0))[0].squeeze(0).squeeze(0) for
            (h, l) in zip(outputs, l)]
        features = torch.stack(features_last)
        # sentence embedding
        batch_size = outputs.size(0)
        batch_ws2 = self.w_s2.expand(batch_size, self.w_s2.size(0), self.w_s2.size(1))
        batch_ws1 = self.w_s1.expand(batch_size, self.w_s1.size(0), self.w_s1.size(1))
        A = self.softmax(torch.bmm(batch_ws2, self.tanh(torch.bmm(batch_ws1, outputs.permute(0, 2, 1)))))
        M = torch.bmm(A, outputs)
        M_logit = self.fc(M)
        features_logit = self.fc(features.unsqueeze(1))
        logit = self.sigmoid(self.fc_f(torch.cat([M_logit, features_logit], dim=1).squeeze(-1)))
        return logit, A

    def predict(self, x, l):
        logit, _ = self.forward(x, l)
        predict_y = torch.tensor([1 if torch.gt(i, 0.5) else 0 for i in logit])
        return predict_y
