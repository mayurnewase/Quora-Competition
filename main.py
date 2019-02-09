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

from utils import *
from preprocessing import *
from tokenizer import *
from modelling import *
from loader import *
from trainer import *

def main():
	"""
	take arguments - > data ,embeedding dir, model_ckpt, which embedding to use
	"""
	parser = argparse.ArgumentParser(description = "feed me properly")
	parser.add_argument("--data_dir")
	parser.add_argument("--embed_dir")
	parser.add_argument("--do_preprocess", type=str, default="True")
	parser.add_argument("--use_extra_features")
	parser.add_argument("--validate", default = True)
	parser.add_argument("--local_validation", type=str, default= "True")
	parser.add_argument("--lower_text")
	parser.add_argument("--max_vocab_words", type=int, default=90000)
	parser.add_argument("--max_seq_len", type = int, default=70)
	parser.add_argument("--filters")
	parser.add_argument("--use_embeddings", default = "False")
	parser.add_argument("--load_glove", default = None)
	parser.add_argument("--load_fast", default = None)
	parser.add_argument("--load_para", default = None)
	parser.add_argument("--single_matrix", default = None)
	parser.add_argument("--mean_matrix", default = None)
	parser.add_argument("--concat_matrix", default = None)
	parser.add_argument("--splits", type = int , default = 2)
	parser.add_argument("--epochs", type = int , default = 5)
	parser.add_argument("--batch_size", type = int , default = 512)

	parser.add_argument("--save_result", type = str, default=True)	
	parser.add_argument("--result_dir", type = str, default=".")	
	args = parser.parse_args()

	
	# Load data
	train = pd.read_csv(args.data_dir+ "train.csv")[:10000]
	test = pd.read_csv(args.data_dir + "test.csv")[:100]
	n_test = len(test) * 3

	X_local_test = None
	Y_local_test = None
	train_feats = np.zeros((train.shape[0], 2))
	test_feats = np.zeros((test.shape[0], 2))
	local_test_feats = np.zeros((n_test, 2))

	if(args.do_preprocess == "True"):
		print("--------preprocessing------------")
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
		Y_local_test = X_local_test.loc[:, "target"].values
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
		print("---------------skipping loading embeddings--------------------")
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

	logger = pd.DataFrame()
	
	print("-----------starting training--------------")
	train_glove, test_glove, local_test_glove = trainer(splits, model, 
		X_train, Y_train , args.epochs, test_loader, local_test_loader
		,train_preds, test_preds, local_test_preds, train_feats, args.batch_size ,args.validate ,args.use_extra_features, logger)

	op = threshold_search(Y_train, train_glove)
	logger.loc[logger.shape[0] - args.epochs , "final_train_f1"] = op["f1"]

	best = metrics.f1_score(Y_local_test, local_test_glove.mean(axis = 1) > op["threshold"])
	logger.loc[logger.shape[0] - args.epochs, "mean_local_test_f1"] = best
	
	s = pd.DataFrame(test_glove).corr()
	a = []
	for i in range(s.shape[0]):
	    for j in range(s.shape[1]):
	        if(i != j):
	            a.append(s.iloc[i,j])
	logger.loc[logger.shape[0] - args.epochs, "test_corr"] = np.mean(a)
	logger.loc[logger.shape[0], :] = "-"
	print(best)

	print(logger)

	if(args.save_result == "True"):
		logger.to_csv(args.result_dir + "/glove_only.csv")
		s.to_csv(args.result_dir + "/glove_only_corr.csv")

if __name__ == "__main__":
	main()


















