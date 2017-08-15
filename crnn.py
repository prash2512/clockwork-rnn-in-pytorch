import numpy as np
import torch.nn as nn
import torch
import torch.nn.parameter
from utils import random, glorotize, orthogonalize
from torch.autograd import Variable
import torch.autograd as grad
from torch.nn import functional as F 


def recurrent_mask(nclocks, nstates):
	matrix = []
	for c in range(nclocks, 0, -1):
		zero_blocks = np.zeros((nstates, nstates * (nclocks - c)))		
		one_blocks = np.ones((nstates, nstates * (c)))
		matrix.append(np.concatenate([zero_blocks, one_blocks], axis=1))
	mask = np.concatenate(matrix, axis=0)
	return mask


def make_schedule(clock_periods, nstates):
	sch = []
	for c in clock_periods:
		for i in range(nstates):
			sch.append(c)
	return sch

class CRNN(nn.Module):
	def __init__(self, dinput, nstates, doutput, clock_periods, full_recurrence=False, learn_state=True, first_layer=False):
		super(CRNN, self).__init__()
		nclocks = len(clock_periods)
		
		Wi = random(nclocks * nstates, dinput + 1)
		Wh = random(nclocks * nstates, nclocks * nstates + 1)
		Wo = random(doutput, nclocks * nstates + 1)
		
		H_0 = np.zeros((nclocks * nstates, 1))

		Wi = glorotize(Wi)
		Wh[:, :-1] = orthogonalize(Wh[:, :-1])
		Wo = glorotize(Wo)

		utri_mask = recurrent_mask(nclocks, nstates)
		if not full_recurrence:
			Wh[:,:-1] *= utri_mask


		schedules = make_schedule(clock_periods, nstates)

		self.dinput = dinput
		self.nstates = nstates
		self.doutput = doutput
		self.clock_periods = clock_periods
		self.nclocks = nclocks
		self.Wi = nn.Parameter(torch.from_numpy(Wi).float())
		self.Wh = nn.Parameter(torch.from_numpy(Wh).float())
		self.Wo = nn.Parameter(torch.from_numpy(Wo).float())
		self.H_0 = torch.from_numpy(H_0).float()
		self.utri_mask = utri_mask
		self.schedules = schedules
		self.full_recurrence = full_recurrence
		self.learn_state = learn_state
		self.first_layer = first_layer
		self.H_last = None

	def forward(self, X):
		
		#input shape of form (seq_length,batch_size,input_dim)
		X = X.transpose(1,2)
		T, n, B = X.size()
		nclocks = self.nclocks
		nstates = self.nstates
		#print "hello"
		Wi = self.Wi
		Wh = self.Wh
		Wo = self.Wo
		
		Ys = Variable(torch.zeros((T, self.doutput, B)))
		
	
		H_prev = Variable(torch.zeros((nclocks * nstates, B)))

		v = torch.cat([self.H_0] * B, 1)

		for t in xrange(T):
			
			val = nclocks * nstates

			active = []
			for i in range(len(self.schedules)):
				active.append(int(t%self.schedules[i]==0))
			
			active = Variable(torch.FloatTensor(active).view(-1,1))

			input = torch.cat([X[t],Variable(torch.ones(1,B))],0)
			
			i_h = torch.mm(Wi,input)
			
			_H_prev = torch.cat([H_prev,Variable(torch.ones((1, B)))],0)
			
			h_h = torch.mm(Wh, _H_prev)
			
			h_new = i_h + h_h

			H_new = F.tanh(h_new)

			H = active.expand_as(H_new)*H_new+(1-active).expand_as(H_prev)*H_prev
			
			_H = torch.cat([H, Variable(torch.ones((1, B)))],0)

			y = torch.mm(Wo, _H)
			
			Y = F.tanh(y)
			
			H_prev = H
			
			Ys[t] = Y

		Ys = Ys.transpose(1,2)
		
		self.H_last = H	
		
		return Ys,H
