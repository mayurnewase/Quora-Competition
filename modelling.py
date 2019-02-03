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
