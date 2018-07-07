
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
os.environ['KERAS_BACKEND'] = 'theano'
from load import *
from metrics.accuracy import conlleval
from load import *
from model import LSTMTagger
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
from accuracy import conlleval
import torch.optim as optim
import progressbar
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)
    #print l
    lpadded = win/2 * [l[0]] + l + win/2 * [l[-1]]
    #print lpadded
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

### Load Data
train_set, valid_set,test_set,dicts = load_dataset(3)
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# get index to word dict
idx2w  = get_id_to_word_dict(dicts)
idx2la = get_id_to_label_dict(dicts)

plt.xlabel('epochs')
plt.ylabel('learning rate')
### Model
n_classes = len(idx2la)
n_vocab = len(idx2w)

model = LSTMTagger(100,100,100,7,n_vocab,n_classes)
loss_function= nn.NLLLoss()
lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr)
#tag_scores = model(Variable(torch.LongTensor([123,343,234,234,234])))

train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
test_x, test_ne, test_label = test_set
validationX = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
validationY = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
trainX = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
trainY = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
testX = [ list(map(lambda x: idx2w[x], w)) for w in test_x]
testY = [ list(map(lambda x: idx2la[x], y)) for y in test_label]

seed = 345
#lr = 0.01
test_loss = []
train_loss = []
validation_loss = []
### Training
n_epochs = 20

train_f_scores = []
val_f_scores = []
test_f_scores = []
best_val_f1 = 0
best_test_f1 = 0

learning =[]
for i in range(n_epochs):

    learning.append(lr)
    print("Epoch {}".format(i))
    #shuffle([train_x, train_ne, train_label],seed)
    print("Training =>")
    train_pred_label = []
    avgLoss = 0    
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
    	
        #cwords = np.array(contextwin(sent,7))
        #print cwords
    	model.zero_grad()
        label = train_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        label = Variable(torch.from_numpy(label).long().view(-1,n_classes).long())
        sent = torch.from_numpy(sent).long().view(-1)
        #print sent
        sent = Variable(torch.LongTensor(sent))
        #print sent.size()
        pred = model(sent)

        #print pred
       	target,index = torch.max(pred,1)

        #print pred[3::7].size()
       	#print index
       	val,index = torch.max(label,1)
        #print index
       	#print index.size()
        #print pred.view(1,-1).size()
        #print label.view(-1).size()
        #print pred.size()
        #print label
        loss = loss_function(pred,index.view(-1))
        avgLoss += loss
        target,index = torch.max(pred,1)
        index = index.transpose(0,1)
        a = index.data.view(-1).numpy().tolist()
        #print a
        train_pred_label.append(a)
        #print loss
        loss.backward()
        optimizer.step()
    #print len(train_pred_label)
    avgLoss = avgLoss/n_batch
    #print avgLoss.data.numpy().astype(int)
    #plt.plot(i,avgLoss,'r')
    #print avgLoss
    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_train, trainY, trainY , 'r.txt')
    train_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))
    train_loss.append(avgLoss.data.numpy()[0])
    print("\n")
    print("Validating =>")
    train_pred_label = []
    avgLoss = 0    
    model.eval()
    bar = progressbar.ProgressBar(max_value=len(val_x))
    for n_batch, sent in bar(enumerate(val_x)):
    	model.zero_grad()

        #cwords = np.array(contextwin(sent,7))
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        label = Variable(torch.from_numpy(label).long().view(-1,n_classes).long())
        sent = torch.from_numpy(sent).long().view(-1)
        sent = Variable(torch.LongTensor(sent))
        #print sent.size()
        pred = model(sent)
       	target,index = torch.max(pred,1)
       	#print index
       	val,index = torch.max(label,1)
       	#print index.size()
        #print pred.view(1,-1).size()
        #print label.view(-1).size()
        #print pred.size()
        #print label
        loss = loss_function(pred,index.view(-1))
        avgLoss += loss
        target,index = torch.max(pred,1)
        index = index.transpose(0,1)
        a = index.data.view(-1).numpy().tolist()
        #print a
        train_pred_label.append(a)
        #print loss
    #print len(train_pred_label)
    avgLoss = avgLoss/n_batch
    #print avgLoss.data.numpy().shape
    validation_loss.append(avgLoss.data.numpy()[0])
    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_val, validationY, validationY , 'r.txt')
    val_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))
    #validation_loss.append(avgLoss.numpy())

    if con_dict['f1'] > best_val_f1:
        best_val_f1 = con_dict['f1']
    
    print("Best validation F1 score = {}".format(best_val_f1))
    print "\n"

    print("Testing =>")
    train_pred_label = []
    avgLoss = 0    
    bar = progressbar.ProgressBar(max_value=len(test_x))
    for n_batch, sent in bar(enumerate(test_x)):
    	model.zero_grad()

        #cwords = np.array(contextwin(sent,7))
        label = test_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        label = Variable(torch.from_numpy(label).long().view(-1,n_classes).long())
        sent = torch.from_numpy(sent).long().view(-1)
        sent = Variable(torch.LongTensor(sent))
        #print sent.size()
        pred = model(sent)
       	target,index = torch.max(pred,1)
       	#print index
       	val,index = torch.max(label,1)
       	#print index.size()
        #print pred.view(1,-1).size()
        #print label.view(-1).size()
        #print pred.size()
        #print label
        loss = loss_function(pred,index.view(-1))
        avgLoss += loss
        target,index = torch.max(pred,1)
        index = index.transpose(0,1)
        a = index.data.view(-1).numpy().tolist()
        #print a
        train_pred_label.append(a)
        #print loss
    #print len(train_pred_label)
    avgLoss = avgLoss/n_batch
    #print avgLoss
    test_loss.append(avgLoss.data.numpy()[0])
    predword_test = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_test, testY, testY , 'r.txt')
    val_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_test_f1:
        best_test_f1 = con_dict['f1']
    else:
        lr = lr/2
    print("Best test F1 score = {}".format(best_test_f1))
    print "\n"

plt.plot(learning,'b')
plt.xlabel("epochs")
plt.ylabel("learing rate")
plt.legend()
plt.savefig("learning.png")
print "best test ",best_test_f1
print "best val. ",best_val_f1       	
