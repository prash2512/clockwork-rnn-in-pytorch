import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from crnn import CRNN

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim,conv_out,contextwin,vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(0.25)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #self.conv = nn.Conv1d(embedding_dim,conv_out,contextwin,padding=2)
        #self.pool = nn.MaxPool1d(contextwin,padding=5)
        self.lstm = CRNN(embedding_dim,hidden_dim,hidden_dim,[1,2,4])
        #self.lstm1 = CRNN(hidden_dim,hidden_dim,hidden_dim,[1,3,9])
        #self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.dim = 6
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        #print sentence.size()
        embeds = self.word_embeddings(sentence)

        #print embeds.size()
        #qn = torch.norm(embeds, p=2, dim=1).detach()
        #embeds = embeds.div(qn.expand_as(embeds))

        #print embeds.size()
        #print embeds.view(len(sentence), 1, -1).size()
        #print embeds.view(len(sentence),1,-1).size()
        #conv_out = F.relu(self.conv(embeds.view(1,-1,len(sentence))))
        #print conv_out.size()
        #conv_out = self.pool(conv_out)
        #print conv_out.size()
        #print embeds.view(len(sentence),1,-1).size()
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence),1,-1))
        #print lstm_out.size()
        #print lstm_out.size()
        #crnn_out,self.hidden = self.crnn(lstm_out)
        #print lstm_out.size()
        #print lstm_out.view(len(sentence), -1).size()
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #print tag_space
        tag_scores = F.log_softmax(tag_space)
        return tag_scores