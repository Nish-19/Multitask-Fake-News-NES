# All general imports
import torch
import transformers as ppb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Bidirectional, GlobalAveragePooling1D, GRU, GlobalMaxPooling1D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPool1D, Activation, Add

from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers.core import SpatialDropout1D

#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation, Conv1D, Softmax
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import io, os, gc

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 
print(dev) 
device = torch.device(dev)  

# Data Loader for embeddings
def get_embeddings(input_id, attention_mask, model, name):
	store = list()
	extra = list()
	length = input_id.shape[0]
	interval_size = 100
	start = 0
	while(start<length):
		print(start)
		if (start+100) < length:
			des = start+100
			pre_batch_in = input_id[start:des,:]
			pre_batch_at = attention_mask[start:des,:]
			with torch.no_grad():
				last_hidden_states = model(pre_batch_in, attention_mask=pre_batch_at)
			store.append(last_hidden_states[0][:,0,:].cpu().numpy())
		else:
			des = length+1
			pre_batch_in = input_id[start:des,:]
			pre_batch_at = attention_mask[start:des,:]
			with torch.no_grad():
			  last_hidden_states = model(pre_batch_in, attention_mask=pre_batch_at)
			extra.append(last_hidden_states[0][:,0,:].cpu().numpy())
		start+=100
#store_np = np.array(store).reshape(length, 768)
	store_np = np.stack(store)
	store_np = store_np.reshape(store_np.shape[0]*store_np.shape[1],768)
	extra_np = np.stack(extra)
	extra_np = extra_np.reshape(extra_np.shape[0]*extra_np.shape[1],768)
	print('store',store_np.shape)
	print('extra',extra_np.shape)
	final_np = np.concatenate([store_np, extra_np], axis=0)
	np.save(name, final_np)
	print(final_np.shape)
	return final_np 

def prepare_bert_embeddings(input_df, save1, save2):
	# For DistilBERT:
	model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

	## Want BERT instead of distilBERT? Uncomment the following line:
	#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

	# Load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights)
	model.to(torch.device(device))

	# Applying tokenization
	tokenized1 = input_df["Headline"].apply((lambda x: tokenizer.encode(x[:510], add_special_tokens=True)))
	tokenized2 = input_df["Body"].apply((lambda x: tokenizer.encode(x[:510], add_special_tokens=True)))

	max_len = 0
	for i in tokenized1.values:
	    if len(i) > max_len:
	        max_len = len(i)

	padded1 = np.array([i + [0]*(max_len-len(i)) for i in tokenized1.values])

	max_len = 0
	for i in tokenized2.values:
	    if len(i) > max_len:
	        max_len = len(i)

	padded2 = np.array([i + [0]*(max_len-len(i)) for i in tokenized2.values])

	print("Premise", np.array(padded1).shape)
	print("Hypothesis", np.array(padded2).shape)

	# Attention masks
	attention_mask1 = np.where(padded1 != 0, 1, 0)
	attention_mask1.shape
	attention_mask2 = np.where(padded2 != 0, 1, 0)
	attention_mask2.shape

	# Creating input ids
	input_ids1 = torch.tensor(padded1).to(device)  
	attention_mask1 = torch.tensor(attention_mask1).to(device)
	input_ids2 = torch.tensor(padded2).to(device)  
	attention_mask2 = torch.tensor(attention_mask2).to(device)

	# Getting the embeddings
	pre_bert = get_embeddings(input_ids1, attention_mask1, model, save1)
	hyp_bert = get_embeddings(input_ids2, attention_mask2, model, save2)
# Importing the datasets (Please chose appropriate folder)
train_df = pd.read_csv('../FNC_Dataset/train_fnc_mt2nd.csv')
print(train_df.columns)
le = LabelEncoder()
# train_df['Stance'] = le.fit_transform(train_df['Stance'])
# train_df.head()

# Test set
test_df = pd.read_csv('../FNC_Dataset/test_fnc_agdgonly.csv')
print(test_df.columns)
# test_df['Stance'] = le.transform(test_df['Stance'])
# test_df.head()

prepare_bert_embeddings(train_df, "fnc/pre_bert_fnc", "fnc/hyp_bert_fnc")
prepare_bert_embeddings(test_df, "fnc/pre_bert_test_fnc", "fnc/hyp_bert_test_fnc")