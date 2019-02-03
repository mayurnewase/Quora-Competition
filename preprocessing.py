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

