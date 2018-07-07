from __future__ import print_function
from os.path import isfile
import numpy as np
import os
import gzip
import six.moves.cPickle as pickle
import urllib
import random
import time
import sys
import stat
import subprocess
import torch

PREFIX = os.getenv('ATISDATA','')

def download_data(filename):
	origin = 'http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/'+filename
	print('Downloading data from %s' % origin)
	name = 'data/'+origin.split('/')[-1]
	urllib.urlretrieve(origin, name)

def load_dataset(fold):
	if fold>5 or fold<1:
		print("enter valid fold")
		return null
	else :
		filename = PREFIX + 'atis.fold'+str(fold)+'.pkl.gz'
		pathname = "data/"+filename
		if not isfile(pathname):
			download_data(filename)	
		print(filename)
		f = gzip.open(pathname,'rb')
		train_set, valid_set, test_set, dicts = pickle.load(f)
		return train_set,valid_set,test_set,dicts

def get_words_from_id(dataset,dicts):
	w2idx = dicts['words2idx']
	sentences = list()
	inverse_mapping = {v: k for k, v in w2idx.iteritems()}
	for i in dataset[0]:
		sentence = list()
		for j in i:
			sentence.append(inverse_mapping[j])
		sentences.append(sentence)
	return sentences

def get_labels_from_id(dataset,dicts):
	l2idx = dicts['labels2idx']
	sentences = list()
	
	inverse_mapping = {v: k for k, v in l2idx.iteritems()}
	for i in dataset[2]:
		sentence = list()
		for j in i:
			sentence.append(inverse_mapping[j])
		sentences.append(sentence)
	return sentences

def get_id_to_word_dict(dicts):
	w2idx = dicts['words2idx']
	inverse_mapping = {v: k for k, v in w2idx.iteritems()}
	return inverse_mapping

def get_id_to_label_dict(dicts):
	l2idx = dicts['labels2idx']
	inverse_mapping = {v: k for k, v in l2idx.iteritems()}
	return inverse_mapping

if __name__ == '__main__':
    #loads data and helps in viudalizing dataset structure
    #words contain actual sentences
    #labels contain slot value that is associated with each word
    print("enter fold value 1-5 for which dataset has to be trained : ")
    fold = int(input())
    train_set,valid_set,test_set,dicts = load_dataset(fold)
    for i in [train_set,valid_set,test_set]:
    	sentences_list = get_words_from_id(i,dicts)
    	labels_list = get_labels_from_id(i,dicts)
    	for j in range(len(sentences_list)):
    		print("sentence")
    		print(sentences_list[j])
    		print("label")
    		print(labels_list[j])
    		print("\n")