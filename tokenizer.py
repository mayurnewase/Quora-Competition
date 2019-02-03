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
	
