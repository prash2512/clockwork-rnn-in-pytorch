from __future__ import print_function
from os.path import isfile
import theano
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
import os
import stat
import subprocess
from os.path import isfile, join
from os import chmod

PREFIX = os.getenv('ATISDATA', '')

def conlleval(p, g, w, filename):
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, w in zip(sl, sp, sw):
            out += w + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()
    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = PREFIX + 'conlleval.pl'
    if not isfile(_conlleval):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl') 
        os.system('wget https://www.comp.nus.edu.sg/%7Ekanmy/courses/practicalNLP_2008/packages/conlleval.pl')
        chmod('conlleval.pl', stat.S_IRWXU) # give the execute permissions

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename,'rb').read())
    for line in stdout.decode("utf-8").split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    
    # out = ['accuracy:', '16.26%;', 'precision:', '0.00%;', 'recall:', '0.00%;', 'FB1:', '0.00']
    
    precision = float(out[3][:-2])
    recall    = float(out[5][:-2])
    f1score   = float(out[7])

    return {'p':precision, 'r':recall, 'f1':f1score}