#-------- COPYING FILE utils.py--------------

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def kmax_pooling(x, dim = 1, k = 3):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index).view(x.shape[0], -1)


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration
        
        self.last_loss = np.inf
        self.min_lr = min_lr
        self.factor = factor
        
    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def step(self, loss):
        if loss > self.last_loss:
            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]
            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]
            
    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs
#-------- COPYING FILE preprocessing.py--------------

"""
input -> dirty dataframe  -> [id, question_text]
output -> clean dataframe
master function
"""
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import math
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler


class preprocess():
	def __init__(self, train, test, use_extra_features, lower_text):

		self.train = train
		self.test = test
		self.train_shape = self.train.shape[0]

		self.test["target"] = 1000
		self.combo = pd.concat([self.train, self.test],axis = 0)

		self.use_extra_features = use_extra_features
		self.extra_features = None
		self.lower_text = lower_text

	def LowerText(self):
		self.combo["question_text"] = self.combo["question_text"].apply(lambda x: x.lower())


	def ReplacePunctuations(self):
		replacers = {"’" : "'", "”" : "\"",
            "’":"'", "”":"\"", "“": "\"", "’" : "'", "”" : "\"", "“" : "\"", "++": "+", "...": ".", "…":"."}
		def replace_punct(x):
		    x = str(x)
		    for punct in replacers:
		        x = x.replace(punct, replacers[punct])
		    return x

		self.combo["question_text"] = self.combo["question_text"].apply(lambda x: replace_punct(x))


	def CorrectMisspells(self):

		mispell_dict = {"i’m":"I am", "what's":"what is","don’t":"do not","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

		def _get_mispell(mispell_dict):
		    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
		    return mispell_dict, mispell_re
		def replace_typical_misspell(text):
		    def replace(match):
		        return mispellings[match.group(0)]
		    return mispellings_re.sub(replace, text)
		
		mispellings, mispellings_re = _get_mispell(mispell_dict)
		self.combo["question_text"] = self.combo["question_text"].apply(lambda x: replace_typical_misspell(x))


	def CorrectSpacing(self):

		space_dict = ["?", ",","\"", "(", ")", "'", "%", "[", "]", "$", "/", ":", ".", "^", "-", "+", "#"]
		def space_punct(x):
			for punct in space_dict:
				x = x.replace(punct, f' {punct} ')
			return x

		self.combo["question_text"] = self.combo["question_text"].apply(lambda x: space_punct(x))


	def AddExtraFeatures(self):

		df = self.combo.copy()

		df['question_text'] = df['question_text'].apply(lambda x:str(x))
		df['total_length'] = df['question_text'].apply(len)
		df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
		df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
		df['caps_vs_length'] = df['caps_vs_length'].fillna(0)
		df['num_words'] = df.question_text.str.count('\S+')
		df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
		df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
		df['words_vs_unique'] = df['words_vs_unique'].fillna("##")
		df["caps_vs_length"] = df["caps_vs_length"]
		ss = StandardScaler()
		self.extra_features = ss.fit_transform(df[["words_vs_unique", "caps_vs_length"]])

	def FullPreprocessing(self):

		if(self.lower_text):
			self.LowerText()

		self.ReplacePunctuations()
		self.CorrectMisspells()
		self.CorrectSpacing()
		
		train_df = self.combo.iloc[:self.train_shape, :]
		test_df = self.combo.iloc[self.train_shape:, :]
		test_df = test_df.drop(["target"], axis = 1)

		if(self.AddExtraFeatures):
			self.AddExtraFeatures()
			train_feats = self.extra_features[:self.train_shape]
			test_feats = self.extra_features[self.train_shape:]
			return [train_df, test_df, train_feats, test_feats]

		else:
			return train_df, test_df, np.zeros((train_df.shape[0], 2)), np.zeros((test_df.shape[0], 2))

#-------- COPYING FILE tokenizer.py--------------

"""
Encode and pad sequences

input -> normal series
output -> encoded array, vocab
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class TokenizerBase():
	def __init__(self, X_train, X_test, X_local_test, lower, filters, num_vocab_words, max_seq_len):

		self.X_train = X_train
		self.X_test = X_test
		self.X_local_test = X_local_test
		self.lower = lower
		self.filters = filters
		self.num_vocab_words = num_vocab_words
		self.max_seq_len = max_seq_len

	def KerasTokenizer(self):

		tokenizer = Tokenizer(num_words = self.num_vocab_words, lower = self.lower, filters = self.filters)
		tokenizer.fit_on_texts(list(self.X_train))
		self.X_train = tokenizer.texts_to_sequences(self.X_train)
		self.X_test = tokenizer.texts_to_sequences(self.X_test)
		if self.X_local_test is not None:
			self.X_local_test = tokenizer.texts_to_sequences(self.X_local_test)

		self.vocab = tokenizer.word_index

	def PadSequence(self):

		self.X_train = pad_sequences(self.X_train, maxlen = self.max_seq_len)
		self.X_test = pad_sequences(self.X_test, maxlen = self.max_seq_len)
		if self.X_local_test is not None:
			self.X_local_test = pad_sequences(self.X_local_test, maxlen=self.max_seq_len)


	def FullTokenizer(self):

		self.KerasTokenizer()
		self.PadSequence()

		return [self.X_train, self.X_test, self.X_local_test, self.vocab]
	
#-------- COPYING FILE modelling.py--------------

"""
load embedding
create models

input -> embedding directory			
output -> models, embedding
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils import *

class Embedder():
	def __init__(self, embedding_directory, model_weights_directory = None):
		
		self.model_weights_directory = model_weights_directory

	def load_embed(file, name):
	    def get_coefs(word,*arr): 
	        return word, np.asarray(arr, dtype='float32')
	    
	    if name == "fast":
	        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
	    else:
	        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
	    return embeddings_index


	def LoadIndexFile(self, load_glove, load_fast, load_para, embedding_directory):

		if(load_glove):
			self.glove_index = load_embed(embedding_directory + "glove.840B.300d/glove.840B.300d.txt", "glove")
		if(load_para):
			self.para_index = load_embed(embedding_directory + "paragram_300_sl999/paragram_300_sl999.txt", "para")
		if(load_fast):
			self.fast_index = load_embed(embedding_directory + "wiki-news-300d-1M/wiki-news-300d-1M.vec", "fast")

	def get_matrix(self, embeddings_index, vocab):
	    all_embs = np.stack(embeddings_index.values())
	    emb_mean,emb_std = all_embs.mean(), all_embs.std()
	    embed_size = all_embs.shape[1]
	    nb_words = min(max_features, len(vocab))
	    #embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
	    embedding_matrix = np.zeros((nb_words, embed_size))
	    for word, i in vocab.items():
	        if i >= max_features: continue
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	        	embedding_matrix[i] = embedding_vector
	    return embedding_matrix


	def GetSingleMatrix(self, name, vocab, embed_size = 300):
		self.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}
		embed = self.embed_name_dict.get(name)
		embed = get_matrix(embed, vocab)
		return embed

	def GetConcatMatrix(self, embed_1, embed_2, vocab, use_features = False, embed_size = 300):

		self.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}

		embed_1 = self.embed_name_dict.get(embed_1)
		embed_2 = self.embed_name_dict.get(embed_2)

		embed_1_some = embed_1["something"]
		embed_2_some = embed_2["something"]
		if use_features:
		    s2 = 301
		else:
		    s2 = 300

		concat_matrix = np.zeros((max_features, embed_size + s2))
		embed_some = np.zeros((embed_size + s2, ))
		embed_some[0:300] = embed_1_some
		embed_some[s2: s2+embed_size] = embed_2_some

		if use_features:
		    embed_some[300] = 0

		def embed_word(embed_matrix_1, embed_matrix_2, vocab_index, word):
		    vector_1 = embed_matrix_1.get(word)
		    concat_matrix[vocab_index, :300] = vector_1
		    
		    vector_2 = embed_matrix_2.get(word)
		    if vector_2 is not None:
		            concat_matrix[vocab_index, s2:] = vector_2

		    if(word.isupper() and len(word) > 1 and use_features):
		        concat_matrix[300] = 1

		for word,index in vocab.items():
		    if index >= max_features:
		        continue
		    glove_vector = embed_1.get(word)
		    if glove_vector is not None:
		        embed_word(embed_1,embed_2, index, word)
		    else:
		        concat_matrix[index] = embed_some
		return concat_matrix

	def GetMeanMatrix(self, embed_1, embed_2, vocab, embed_siz = 300):

		self.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}
		embed_1 = self.embed_name_dict.get(embed_1)
		embed_2 = self.embed_name_dict.get(embed_2)

		mat1 = get_matrix(embed_1, vocab)
		mat2 = get_matrix(embed_2, vocab)
		mean_mat = np.mean([mat1, mat2], axis = 0)
		del mat1, mat2

		return mean_mat


class Model(nn.Module):
	def __init__(self, embed_size,max_features ,maxlen ,embedding_matrix = None):
		super(Model, self).__init__()

		hidden_size = 128

		self.embedding1 = nn.Embedding(max_features, embed_size)
		if(embedding_matrix != None):
			self.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
		self.embedding1.weight.requires_grad = False

		self.embedding_dropout = nn.Dropout2d(0.2)

		self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
		self.gru1 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
		self.lstm_attention1 = Attention(hidden_size*2, maxlen)
		self.gru_attention1 = Attention(hidden_size*2, maxlen)

		self.linear1 = nn.Linear(1536, 32)
		self.linear2 = nn.Linear(32,1)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(0.2)
        
	def forward(self, x, _):
		h_embedding1 = self.embedding1(x)
		h_embedding1 = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding1, 0)))

		h_lstm1, _ = self.lstm1(h_embedding1)
		h_gru1, _ = self.gru1(h_lstm1)
		h_lstm_atten1 = self.lstm_attention1(h_lstm1)
		h_gru_atten1 = self.gru_attention1(h_gru1)
		avg_pool1 = torch.mean(h_gru1, 1)
		ktop_pool1 = kmax_pooling(h_gru1, k = 3)
		#max_pool1, _ = torch.max(h_gru1, 1)
		conc = torch.cat((h_lstm_atten1, h_gru_atten1, avg_pool1, ktop_pool1), 1)
		conc1 = self.dropout(self.relu(self.linear1(conc)))
		op = self.linear2(conc1)

		return op


class ModelWithFeats(nn.Module):
	def __init__(self, embed_size, max_features, maxlen, embedding_matrix = None):
		super(ModelWithFeats, self).__init__()

		hidden_size = 128

		self.embedding1 = nn.Embedding(max_features, embed_size)
		if(embedding_matrix != None):
			self.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
		self.embedding1.weight.requires_grad = False

		self.embedding_dropout = nn.Dropout2d(0.2)

		self.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
		self.gru1 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)
		self.lstm_attention1 = Attention(hidden_size*2, maxlen)
		self.gru_attention1 = Attention(hidden_size*2, maxlen)

		self.linear1 = nn.Linear(1538, 32)
		self.linear2 = nn.Linear(32,1)
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(0.2)
        
	def forward(self, x, feats):
		h_embedding1 = self.embedding1(x)
		h_embedding1 = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding1, 0)))

		h_lstm1, _ = self.lstm1(h_embedding1)
		h_gru1, _ = self.gru1(h_lstm1)
		h_lstm_atten1 = self.lstm_attention1(h_lstm1)
		h_gru_atten1 = self.gru_attention1(h_gru1)
		avg_pool1 = torch.mean(h_gru1, 1)
		ktop_pool1 = kmax_pooling(h_gru1, k = 3)
		#max_pool1, _ = torch.max(h_gru1, 1)
		conc = torch.cat((h_lstm_atten1, h_gru_atten1, avg_pool1, ktop_pool1, feats), 1)
		conc1 = self.dropout(self.relu(self.linear1(conc)))
		op = self.linear2(conc1)

		return op
#-------- COPYING FILE loader.py--------------

import torch.utils.data
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_local_loader = None

    x_test_tensor = torch.tensor(test_X ,dtype = torch.long).to(device)
    test_feats_tensor = torch.tensor(test_feats, dtype = torch.float32).to(device)
    x_test_dataset = torch.utils.data.TensorDataset(x_test_tensor, test_feats_tensor)
    test_loader = torch.utils.data.DataLoader(x_test_dataset, batch_size = batch_size*2, shuffle = False)   

    if local_test_X is not None:
    	x_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)

    	x_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)
    	local_test_feats_tensor = torch.tensor(local_test_feats, dtype = torch.float32).to(device)
    	x_local_test_dataset = torch.utils.data.TensorDataset(x_test_local_tensor,local_test_feats_tensor)

    	test_local_loader = torch.utils.data.DataLoader(x_local_test_dataset, batch_size = batch_size*2, shuffle = False)

    return test_loader, test_local_loader


def GetData(test_X, local_test_X, train_X ,train_Y, test_feats, local_test_feats, n_splits, batch_size ,use_extra_features= False):
    test_loader, local_test_loader = GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features)
    print(train_X.shape, test_X.shape)
    if n_splits > 1:
        splits = list(StratifiedKFold(n_splits = n_splits, shuffle=True, random_state= 165).split(train_X, train_Y))
    else:
        valid_index = 30000
        splits = [[np.arange(start = 0, stop = train_X.shape[0]-valid_index), np.arange(start = train_X.shape[0] - valid_index, stop = train_X.shape[0])]]
    train_preds = np.zeros((train_X.shape[0], ))
    test_preds = np.zeros((test_X.shape[0], len(splits)))

    local_test_preds = None
    if(local_test_X is not None):
    	local_test_preds = np.zeros((local_test_X.shape[0], len(splits)))
    
    return test_loader, local_test_loader, splits, train_preds, test_preds, local_test_preds

#-------- COPYING FILE trainer.py--------------

"""
Model training methods
"""
import torch.utils.data
import torch
import numpy as np
import copy
from utils import *
import time
from tqdm import tqdm
from sklearn import metrics

def train_model(model,folds_list, test_loader, local_test_loader, n_epochs, batch_size ,validate, use_extra_features):
    print("\n --------training model----------")
    optimizer = torch.optim.Adam(model.parameters())
    
    step_size = 300
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,
                         step_size=step_size, mode='triangular2',
                         gamma=0.99994)
    
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    #l2_loss = torch.nn.MSELoss().cuda()
    #folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]
    train = torch.utils.data.TensorDataset(folds_list[0],folds_list[4] , folds_list[2])
    valid = torch.utils.data.TensorDataset(folds_list[1], folds_list[5] , folds_list[3])
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size*2, shuffle=False)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        
        for (x_full_batch_train, feat_train ,y_batch_train) in tqdm(train_loader, disable = True):
            y_pred_train = model(x_full_batch_train, feat_train)
            scheduler.batch_step()
            loss = binary_cross_entropy(y_pred_train, y_batch_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        valid_preds = np.zeros((folds_list[1].size(0)))
        if validate == "True":
            avg_val_loss = 0.
            for i, (x_full_batch_val, feat_val ,y_batch_val) in enumerate(valid_loader):
                y_pred_val = model(x_full_batch_val, feat_val).detach()
                avg_val_loss += binary_cross_entropy(y_pred_val, y_batch_val).item() / len(valid_loader)
                valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]
            
            search_result = threshold_search(folds_list[3].cpu().numpy(), valid_preds)
            val_f1, val_threshold = search_result['f1'], search_result['threshold']
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} best_t={:.2f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))
    
    model.eval()
    valid_preds = np.zeros((folds_list[1].size(0)))
    avg_val_loss = 0.
    for i, (x_full_batch_val, feat_val,y_batch_val) in enumerate(valid_loader):
        y_pred_val = model(x_full_batch_val, feat_val).detach()
        avg_val_loss += binary_cross_entropy(y_pred_val , y_batch_val).item() / len(valid_loader)
        valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]
    print('Validation loss: ', avg_val_loss)

    test_preds = np.zeros((len(test_loader.dataset)))
    for i, (x_full_batch_test, feat_test) in enumerate(test_loader):
        y_pred_test = model(x_full_batch_test, feat_test).detach()
        test_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_test.cpu().numpy())[:, 0]
    
    test_preds_local = np.zeros((len(local_test_loader.dataset)))
    if(local_test_loader is not None):   
	    for i, (x_full_batch_local, feat_local) in enumerate(local_test_loader):
	        y_pred_local = model(x_full_batch_local, feat_local).detach()
	        test_preds_local[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_local.cpu().numpy())[:, 0]

    return valid_preds, test_preds, test_preds_local

def trainer(splits, model_orig , train_X, train_Y, epochs, test_loader, local_test_loader ,train_preds, test_preds, local_test_preds, train_feat, batch_size,validate ,use_extra_features):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n ---------splitting----------")
    for split_no, (train_idx, valid_idx) in enumerate(splits):
        
        X_train_fold = torch.tensor(train_X[train_idx], dtype = torch.long).to(device)
        Y_train_fold = torch.tensor(train_Y[train_idx, np.newaxis], dtype = torch.float32).to(device)
        X_val_fold = torch.tensor(train_X[valid_idx], dtype = torch.long).to(device)
        Y_val_fold =torch.tensor(train_Y[valid_idx, np.newaxis], dtype = torch.float32).to(device)
        
        
        train_feat_fold = torch.tensor(train_feat[train_idx], dtype = torch.float32).to(device)
        valid_feat_fold = torch.tensor(train_feat[valid_idx], dtype = torch.float32).to(device)
        
        folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]
        
        model = copy.deepcopy(model_orig)
        model.to(device)
        print("Split {}/{}".format(split_no+1,len(splits)))
        pred_val_fold, pred_test_fold, pred_local_test_fold = train_model(model, folds_list,
                                                                          test_loader, local_test_loader,
                                                                           epochs , batch_size, validate, use_extra_features)
        
        train_preds[valid_idx] = pred_val_fold
        test_preds[:, split_no] = pred_test_fold
        local_test_preds[:, split_no] = pred_local_test_fold
    return train_preds, test_preds, local_test_preds



import argparse

parser = argparse.ArgumentParser() 
parser.add_argument("--data_dir", default = "data/")
parser.add_argument("--embed_dir", default =None)
parser.add_argument("--do_preprocess", default =False)
parser.add_argument("--use_extra_features", default =False)
parser.add_argument("--validate", default =True)
parser.add_argument("--local_validation", default =True)
parser.add_argument("--lower_text", default =True)
parser.add_argument("--max_vocab_words", type=int, default=90000)
parser.add_argument("--max_seq_len", type =int, default=70)
parser.add_argument("--filters", default ="")
parser.add_argument("--use_embeddings", default =False)
parser.add_argument("--load_glove", default =None)
parser.add_argument("--load_fast", default =None)
parser.add_argument("--load_para", default =None)
parser.add_argument("--single_matrix", default =None)
parser.add_argument("--mean_matrix", default =None)
parser.add_argument("--concat_matrix", default =None)
parser.add_argument("--splits", type = int , default =2)
parser.add_argument("--epochs", type = int , default =5)
parser.add_argument("--batch_size", type = int , default =128)
args = parser.parse_args()

#-------- COPYING MAIN + --------------

"""
Runs Pipeline
	preprocessing
	tokenize
	modelling
	training
	postprocessing
"""
import os
import numpy as np
import pandas as pd
import numpy as np
import random
import operator
import argparse


def main():
	"""
	take arguments - > data ,embeedding dir, model_ckpt, which embedding to use
	"""

	
	# Load data
	train = pd.read_csv(args.data_dir+ "/train.csv")[:10000]
	test = pd.read_csv(args.data_dir + "/test.csv")[:100]
	n_test = len(test) * 3

	X_local_test = None
	Y_local_test = None
	train_feats = np.zeros((train.shape[0], 1))
	test_feats = np.zeros((test.shape[0], 1))
	local_test_feats = np.zeros((n_test, 1))

	if(args.do_preprocess == "True"):
		preproc = preprocess(train, test, args.use_extra_features, 
			args.lower_text)
		train, test, train_feats, test_feats = preproc.FullPreprocessing()
	print("after preprocess ", train.shape, test.shape)
	#Local Validation
	if (args.local_validation == "True"):
		print("--------preparing cross_validation------------")
		temp = train.copy()
		train = temp.iloc[:-n_test]
		X_local_test = temp.iloc[-n_test:]
		del temp
		y_local_test = X_local_test.loc[:, "target"].values
		print("in local_val ", train.shape, X_local_test.shape)
		temp = train_feats.copy()
		train_feats = temp[:-n_test]
		local_test_feats = temp[-n_test:]
		del temp
		#train_feats, local_test_feats = (train_feats[:-n_test],
	    #	                            train_feats[-n_test:])

	Y_train = train.loc[:, "target"].values

	#Tokenizer
	print("-------tokenizing------------")
	#print("to tokenizer ", train.shape, test.shape, X_local_test.shape)
	if(X_local_test is not None):
		tok = TokenizerBase(train["question_text"], 
			test["question_text"], X_local_test["question_text"], args.lower_text,
			args.filters, args.max_vocab_words, args.max_seq_len)
	else:
		tok = TokenizerBase(train["question_text"], 
			test["question_text"], None, args.lower_text,
			args.filters, args.max_vocab_words, args.max_seq_len)
	X_train, X_test, X_local_test, vocab = tok.FullTokenizer()


	#Embedding and Modelling
	if (args.use_embeddings == "True"):
		print("-------------loading embeddings----------------")
		print(args.use_embeddings)
		embedder = Embedder(args.embed_dir)
		embedder.LoadIndexFile(args.load_glove, args.load_fast, args.load_para, args.embed_dir)
		
		if(args.single_matrix):
			single_matrix = embedder.GetSingleMatrix(args.single_matrix, vocab)
			if(args.use_extra_features == "True"):
				model = ModelWithFeats(300, args.max_vocab_words, args.max_seq_len , single_matrix)
			else:
				model = Model(300, single_matrix)
		elif(args.concat_matrix == "True"):
			embed1, embed2 = args.concat_matrix.split()
			concat_matrix = embedder.GetConcatMatrix(embed1, embed2, vocab)
			if(args.use_extra_features):
				model = ModelWithFeats(300,args.max_vocab_words, args.max_seq_len, concat_matrix)
			else:
				model = Model(601,args.max_vocab_words, args.max_seq_len ,concat_matrix)
		elif(args.mean_matrix == "True"):
			embed1, embed2 = args.mean_matrix.split()
			mean_matrix = embedder.GetMeanMatrix(embed1, embed2, vocab)
			if(args.use_extra_features == "True"):
				model = ModelWithFeats(300,args.max_vocab_words, args.max_seq_len ,mean_matrix)
			else:				
				model = Model(300, args.max_vocab_words, args.max_seq_len ,mean_matrix)
	else:
		print("not loading embeddings")
		if(args.use_extra_features == "True"):
				model = ModelWithFeats(300, args.max_vocab_words, args.max_seq_len)
		else:
			model = Model(300,args.max_vocab_words, args.max_seq_len)

	#Load Data
	#getData(test_X, local_test_X, train_X ,train_Y, test_feats, lcoal_test_feats , n_splits = 3)
	#print(X_test.shape, X_local_test.shape, X_train.shape, Y_train.shape, test_feats.shape)
	print("-----------generating data-------------")
	test_loader, local_test_loader, splits, train_preds, test_preds, local_test_preds = GetData(X_test, X_local_test, 
		X_train, Y_train, test_feats,local_test_feats , args.splits, args.batch_size , args.use_extra_features)
	
	print("-----------starting training--------------")
	train_glove, test_glove, local_test_glove = trainer(splits, model, 
		X_train, Y_train , args.epochs, test_loader, local_test_loader
		,train_preds, test_preds, local_test_preds, train_feats, args.batch_size ,args.validate ,args.use_extra_features)

	op = threshold_search(train_Y, train_glove)
	print(op["f1"], op["threshold"])
	best = metrics.f1_score(local_test_Y, local_test_glove.mean(axis = 1) > op["threshold"])
	print(best)













if __name__ == "__main__":
	main()


















