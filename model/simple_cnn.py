import torch
import torch.nn as nn

from conf import AppConf


class SimpleCNN(nn.Module):
    def __init__(self, vocab_size, class_size, embedding_dim, embedding_matrix, device):
        super(SimpleCNN, self).__init__()
        self.max_sen = AppConf.max_sen
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            weight = torch.from_numpy(embedding_matrix).type(torch.FloatTensor).to(device)
            self.emb = nn.Embedding.from_pretrained(weight)
        self.emb.weight.requires_grad = True
        self.out_channel = 100
        self.kernel_size = 3
        self.sen_num =100
        self.cnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.sen_num, self.sen_num, kernel_size=self.kernel_size ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(AppConf.max_sen - self.kernel_size + 1))
        )
        self.fc = nn.Linear(600, class_size)
        # self.fc = nn.Linear(self.out_channel, class_size)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def get_features(self, x, l):
        emb = self.emb(x)
        emb = emb.permute(0, 1, 3, 2)
        cnn = self.cnn(emb)
        return cnn

    def forward(self, x, l):
        cnn = self.get_features(x, l)
        logit = self.sigmoid(self.fc(cnn.view(cnn.size(0), -1)))
        return logit

    def predict(self, x, l):
        logit = self.forward(x, l)
        predict_y = torch.tensor([1 if torch.gt(i, 0.5) else 0 for i in logit])
        return predict_y

    def compute_loss(self, logit, y):
        loss = self.loss(logit, y)
        return loss
