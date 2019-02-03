#!/usr/bin/env python


import contextlib as __stickytape_contextlib

@__stickytape_contextlib.contextmanager
def __stickytape_temporary_dir():
    import tempfile
    import shutil
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with __stickytape_temporary_dir() as __stickytape_working_dir:
    def __stickytape_write_module(path, contents):
        import os, os.path, errno

        def make_package(path):
            parts = path.split("/")
            partial_path = __stickytape_working_dir
            for part in parts:
                partial_path = os.path.join(partial_path, part)
                if not os.path.exists(partial_path):
                    os.mkdir(partial_path)
                    open(os.path.join(partial_path, "__init__.py"), "w").write("\n")
                    
        make_package(os.path.dirname(path))
        
        full_path = os.path.join(__stickytape_working_dir, path)
        with open(full_path, "w") as module_file:
            module_file.write(contents)

    import sys as __stickytape_sys
    __stickytape_sys.path.insert(0, __stickytape_working_dir)

    __stickytape_write_module('''utils.py''', '''import torch.nn.functional as F\nimport torch\nimport torch.nn as nn\nimport numpy as np\nfrom sklearn import metrics\n\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\ndef threshold_search(y_true, y_proba):\n    best_threshold = 0\n    best_score = 0\n    for threshold in [i * 0.01 for i in range(100)]:\n        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)\n        if score > best_score:\n            best_threshold = threshold\n            best_score = score\n    search_result = {'threshold': best_threshold, 'f1': best_score}\n    return search_result\n\ndef f1(y_true, y_pred):\n    def recall(y_true, y_pred):\n        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))\n        recall = true_positives / (possible_positives + K.epsilon())\n        return recall\n\n    def precision(y_true, y_pred):\n        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))\n        precision = true_positives / (predicted_positives + K.epsilon())\n        return precision\n    precision = precision(y_true, y_pred)\n    recall = recall(y_true, y_pred)\n    return 2*((precision*recall)/(precision+recall+K.epsilon()))\ndef kmax_pooling(x, dim = 1, k = 3):\n    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]\n    return x.gather(dim, index).view(x.shape[0], -1)\n\n\nclass Attention(nn.Module):\n    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):\n        super(Attention, self).__init__(**kwargs)\n        \n        self.supports_masking = True\n\n        self.bias = bias\n        self.feature_dim = feature_dim\n        self.step_dim = step_dim\n        self.features_dim = 0\n        \n        weight = torch.zeros(feature_dim, 1)\n        nn.init.xavier_uniform_(weight)\n        self.weight = nn.Parameter(weight)\n        \n        if bias:\n            self.b = nn.Parameter(torch.zeros(step_dim))\n        \n    def forward(self, x, mask=None):\n        feature_dim = self.feature_dim\n        step_dim = self.step_dim\n\n        eij = torch.mm(\n            x.contiguous().view(-1, feature_dim), \n            self.weight\n        ).view(-1, step_dim)\n        \n        if self.bias:\n            eij = eij + self.b\n            \n        eij = torch.tanh(eij)\n        a = torch.exp(eij)\n        \n        if mask is not None:\n            a = a * mask\n\n        a = a / torch.sum(a, 1, keepdim=True) + 1e-10\n\n        weighted_input = x * torch.unsqueeze(a, -1)\n        return torch.sum(weighted_input, 1)\n\nclass CyclicLR(object):\n    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,\n                 step_size=2000, factor=0.6, min_lr=1e-4, mode='triangular', gamma=1.,\n                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):\n\n        if not isinstance(optimizer, torch.optim.Optimizer):\n            raise TypeError('{} is not an Optimizer'.format(\n                type(optimizer).__name__))\n        self.optimizer = optimizer\n\n        if isinstance(base_lr, list) or isinstance(base_lr, tuple):\n            if len(base_lr) != len(optimizer.param_groups):\n                raise ValueError("expected {} base_lr, got {}".format(\n                    len(optimizer.param_groups), len(base_lr)))\n            self.base_lrs = list(base_lr)\n        else:\n            self.base_lrs = [base_lr] * len(optimizer.param_groups)\n\n        if isinstance(max_lr, list) or isinstance(max_lr, tuple):\n            if len(max_lr) != len(optimizer.param_groups):\n                raise ValueError("expected {} max_lr, got {}".format(\n                    len(optimizer.param_groups), len(max_lr)))\n            self.max_lrs = list(max_lr)\n        else:\n            self.max_lrs = [max_lr] * len(optimizer.param_groups)\n\n        self.step_size = step_size\n\n        if mode not in ['triangular', 'triangular2', 'exp_range'] \\\n                and scale_fn is None:\n            raise ValueError('mode is invalid and scale_fn is None')\n\n        self.mode = mode\n        self.gamma = gamma\n\n        if scale_fn is None:\n            if self.mode == 'triangular':\n                self.scale_fn = self._triangular_scale_fn\n                self.scale_mode = 'cycle'\n            elif self.mode == 'triangular2':\n                self.scale_fn = self._triangular2_scale_fn\n                self.scale_mode = 'cycle'\n            elif self.mode == 'exp_range':\n                self.scale_fn = self._exp_range_scale_fn\n                self.scale_mode = 'iterations'\n        else:\n            self.scale_fn = scale_fn\n            self.scale_mode = scale_mode\n\n        self.batch_step(last_batch_iteration + 1)\n        self.last_batch_iteration = last_batch_iteration\n        \n        self.last_loss = np.inf\n        self.min_lr = min_lr\n        self.factor = factor\n        \n    def batch_step(self, batch_iteration=None):\n        if batch_iteration is None:\n            batch_iteration = self.last_batch_iteration + 1\n        self.last_batch_iteration = batch_iteration\n        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):\n            param_group['lr'] = lr\n\n    def step(self, loss):\n        if loss > self.last_loss:\n            self.base_lrs = [max(lr * self.factor, self.min_lr) for lr in self.base_lrs]\n            self.max_lrs = [max(lr * self.factor, self.min_lr) for lr in self.max_lrs]\n            \n    def _triangular_scale_fn(self, x):\n        return 1.\n\n    def _triangular2_scale_fn(self, x):\n        return 1 / (2. ** (x - 1))\n\n    def _exp_range_scale_fn(self, x):\n        return self.gamma**(x)\n\n    def get_lr(self):\n        step_size = float(self.step_size)\n        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))\n        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)\n\n        lrs = []\n        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)\n        for param_group, base_lr, max_lr in param_lrs:\n            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))\n            if self.scale_mode == 'cycle':\n                lr = base_lr + base_height * self.scale_fn(cycle)\n            else:\n                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)\n            lrs.append(lr)\n        return lrs\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
    __stickytape_write_module('''preprocessing.py''', '''"""\ninput -> dirty dataframe  -> [id, question_text]\noutput -> clean dataframe\nmaster function\n"""\nimport os\nimport time\nimport numpy as np\nimport pandas as pd\nfrom tqdm import tqdm\nimport re\nimport math\nfrom sklearn.model_selection import train_test_split, StratifiedKFold\nfrom sklearn import metrics\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.preprocessing.sequence import pad_sequences\nfrom sklearn.preprocessing import StandardScaler\n\n\nclass preprocess():\n\tdef __init__(self, train, test, use_extra_features, lower_text):\n\n\t\tself.train = train\n\t\tself.test = test\n\t\tself.train_shape = self.train.shape[0]\n\n\t\tself.test["target"] = 1000\n\t\tself.combo = pd.concat([self.train, self.test],axis = 0)\n\n\t\tself.use_extra_features = use_extra_features\n\t\tself.extra_features = None\n\t\tself.lower_text = lower_text\n\n\tdef LowerText(self):\n\t\tself.combo["question_text"] = self.combo["question_text"].apply(lambda x: x.lower())\n\n\n\tdef ReplacePunctuations(self):\n\t\treplacers = {"\u2019" : "'", "\u201d" : "\\"",\n            "\u2019":"'", "\u201d":"\\"", "\u201c": "\\"", "\u2019" : "'", "\u201d" : "\\"", "\u201c" : "\\"", "++": "+", "...": ".", "\u2026":"."}\n\t\tdef replace_punct(x):\n\t\t    x = str(x)\n\t\t    for punct in replacers:\n\t\t        x = x.replace(punct, replacers[punct])\n\t\t    return x\n\n\t\tself.combo["question_text"] = self.combo["question_text"].apply(lambda x: replace_punct(x))\n\n\n\tdef CorrectMisspells(self):\n\n\t\tmispell_dict = {"i\u2019m":"I am", "what's":"what is","don\u2019t":"do not","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}\n\n\t\tdef _get_mispell(mispell_dict):\n\t\t    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))\n\t\t    return mispell_dict, mispell_re\n\t\tdef replace_typical_misspell(text):\n\t\t    def replace(match):\n\t\t        return mispellings[match.group(0)]\n\t\t    return mispellings_re.sub(replace, text)\n\t\t\n\t\tmispellings, mispellings_re = _get_mispell(mispell_dict)\n\t\tself.combo["question_text"] = self.combo["question_text"].apply(lambda x: replace_typical_misspell(x))\n\n\n\tdef CorrectSpacing(self):\n\n\t\tspace_dict = ["?", ",","\\"", "(", ")", "'", "%", "[", "]", "$", "/", ":", ".", "^", "-", "+", "#"]\n\t\tdef space_punct(x):\n\t\t\tfor punct in space_dict:\n\t\t\t\tx = x.replace(punct, f' {punct} ')\n\t\t\treturn x\n\n\t\tself.combo["question_text"] = self.combo["question_text"].apply(lambda x: space_punct(x))\n\n\n\tdef AddExtraFeatures(self):\n\n\t\tdf = self.combo.copy()\n\n\t\tdf['question_text'] = df['question_text'].apply(lambda x:str(x))\n\t\tdf['total_length'] = df['question_text'].apply(len)\n\t\tdf['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))\n\t\tdf['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)\n\t\tdf['caps_vs_length'] = df['caps_vs_length'].fillna(0)\n\t\tdf['num_words'] = df.question_text.str.count('\\S+')\n\t\tdf['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))\n\t\tdf['words_vs_unique'] = df['num_unique_words'] / df['num_words']\n\t\tdf['words_vs_unique'] = df['words_vs_unique'].fillna("##")\n\t\tdf["caps_vs_length"] = df["caps_vs_length"]\n\t\tss = StandardScaler()\n\t\tself.extra_features = ss.fit_transform(df[["words_vs_unique", "caps_vs_length"]])\n\n\tdef FullPreprocessing(self):\n\n\t\tif(self.lower_text):\n\t\t\tself.LowerText()\n\n\t\tself.ReplacePunctuations()\n\t\tself.CorrectMisspells()\n\t\tself.CorrectSpacing()\n\t\t\n\t\ttrain_df = self.combo.iloc[:self.train_shape, :]\n\t\ttest_df = self.combo.iloc[self.train_shape:, :]\n\t\ttest_df = test_df.drop(["target"], axis = 1)\n\n\t\tif(self.AddExtraFeatures):\n\t\t\tself.AddExtraFeatures()\n\t\t\ttrain_feats = self.extra_features[:self.train_shape]\n\t\t\ttest_feats = self.extra_features[self.train_shape:]\n\t\t\treturn [train_df, test_df, train_feats, test_feats]\n\n\t\telse:\n\t\t\treturn train_df, test_df, np.zeros((train_df.shape[0], 2)), np.zeros((test_df.shape[0], 2))\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
    __stickytape_write_module('''tokenizer.py''', '''"""\nEncode and pad sequences\n\ninput -> normal series\noutput -> encoded array, vocab\n"""\nfrom keras.preprocessing.text import Tokenizer\nfrom keras.preprocessing.sequence import pad_sequences\n\nclass TokenizerBase():\n\tdef __init__(self, X_train, X_test, X_local_test, lower, filters, num_vocab_words, max_seq_len):\n\n\t\tself.X_train = X_train\n\t\tself.X_test = X_test\n\t\tself.X_local_test = X_local_test\n\t\tself.lower = lower\n\t\tself.filters = filters\n\t\tself.num_vocab_words = num_vocab_words\n\t\tself.max_seq_len = max_seq_len\n\n\tdef KerasTokenizer(self):\n\n\t\ttokenizer = Tokenizer(num_words = self.num_vocab_words, lower = self.lower, filters = self.filters)\n\t\ttokenizer.fit_on_texts(list(self.X_train))\n\t\tself.X_train = tokenizer.texts_to_sequences(self.X_train)\n\t\tself.X_test = tokenizer.texts_to_sequences(self.X_test)\n\t\tif self.X_local_test is not None:\n\t\t\tself.X_local_test = tokenizer.texts_to_sequences(self.X_local_test)\n\n\t\tself.vocab = tokenizer.word_index\n\n\tdef PadSequence(self):\n\n\t\tself.X_train = pad_sequences(self.X_train, maxlen = self.max_seq_len)\n\t\tself.X_test = pad_sequences(self.X_test, maxlen = self.max_seq_len)\n\t\tif self.X_local_test is not None:\n\t\t\tself.X_local_test = pad_sequences(self.X_local_test, maxlen=self.max_seq_len)\n\n\n\tdef FullTokenizer(self):\n\n\t\tself.KerasTokenizer()\n\t\tself.PadSequence()\n\n\t\treturn [self.X_train, self.X_test, self.X_local_test, self.vocab]\n\t\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
    __stickytape_write_module('''modelling.py''', '''"""\nload embedding\ncreate models\n\ninput -> embedding directory\t\t\t\noutput -> models, embedding\n"""\nimport torch.nn.functional as F\nimport torch\nimport torch.nn as nn\nfrom utils import *\n\nclass Embedder():\n\tdef __init__(self, embedding_directory, model_weights_directory = None):\n\t\t\n\t\tself.model_weights_directory = model_weights_directory\n\n\tdef load_embed(file, name):\n\t    def get_coefs(word,*arr): \n\t        return word, np.asarray(arr, dtype='float32')\n\t    \n\t    if name == "fast":\n\t        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)\n\t    else:\n\t        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))\n\t    return embeddings_index\n\n\n\tdef LoadIndexFile(self, load_glove, load_fast, load_para, embedding_directory):\n\n\t\tif(load_glove):\n\t\t\tself.glove_index = load_embed(embedding_directory + "glove.840B.300d/glove.840B.300d.txt", "glove")\n\t\tif(load_para):\n\t\t\tself.para_index = load_embed(embedding_directory + "paragram_300_sl999/paragram_300_sl999.txt", "para")\n\t\tif(load_fast):\n\t\t\tself.fast_index = load_embed(embedding_directory + "wiki-news-300d-1M/wiki-news-300d-1M.vec", "fast")\n\n\tdef get_matrix(self, embeddings_index, vocab):\n\t    all_embs = np.stack(embeddings_index.values())\n\t    emb_mean,emb_std = all_embs.mean(), all_embs.std()\n\t    embed_size = all_embs.shape[1]\n\t    nb_words = min(max_features, len(vocab))\n\t    #embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))\n\t    embedding_matrix = np.zeros((nb_words, embed_size))\n\t    for word, i in vocab.items():\n\t        if i >= max_features: continue\n\t        embedding_vector = embeddings_index.get(word)\n\t        if embedding_vector is not None:\n\t        \tembedding_matrix[i] = embedding_vector\n\t    return embedding_matrix\n\n\n\tdef GetSingleMatrix(self, name, vocab, embed_size = 300):\n\t\tself.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}\n\t\tembed = self.embed_name_dict.get(name)\n\t\tembed = get_matrix(embed, vocab)\n\t\treturn embed\n\n\tdef GetConcatMatrix(self, embed_1, embed_2, vocab, use_features = False, embed_size = 300):\n\n\t\tself.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}\n\n\t\tembed_1 = self.embed_name_dict.get(embed_1)\n\t\tembed_2 = self.embed_name_dict.get(embed_2)\n\n\t\tembed_1_some = embed_1["something"]\n\t\tembed_2_some = embed_2["something"]\n\t\tif use_features:\n\t\t    s2 = 301\n\t\telse:\n\t\t    s2 = 300\n\n\t\tconcat_matrix = np.zeros((max_features, embed_size + s2))\n\t\tembed_some = np.zeros((embed_size + s2, ))\n\t\tembed_some[0:300] = embed_1_some\n\t\tembed_some[s2: s2+embed_size] = embed_2_some\n\n\t\tif use_features:\n\t\t    embed_some[300] = 0\n\n\t\tdef embed_word(embed_matrix_1, embed_matrix_2, vocab_index, word):\n\t\t    vector_1 = embed_matrix_1.get(word)\n\t\t    concat_matrix[vocab_index, :300] = vector_1\n\t\t    \n\t\t    vector_2 = embed_matrix_2.get(word)\n\t\t    if vector_2 is not None:\n\t\t            concat_matrix[vocab_index, s2:] = vector_2\n\n\t\t    if(word.isupper() and len(word) > 1 and use_features):\n\t\t        concat_matrix[300] = 1\n\n\t\tfor word,index in vocab.items():\n\t\t    if index >= max_features:\n\t\t        continue\n\t\t    glove_vector = embed_1.get(word)\n\t\t    if glove_vector is not None:\n\t\t        embed_word(embed_1,embed_2, index, word)\n\t\t    else:\n\t\t        concat_matrix[index] = embed_some\n\t\treturn concat_matrix\n\n\tdef GetMeanMatrix(self, embed_1, embed_2, vocab, embed_siz = 300):\n\n\t\tself.embed_name_dict = {"glove" : self.glove_index, "fast": self.fast_index, "para" : self.para_index}\n\t\tembed_1 = self.embed_name_dict.get(embed_1)\n\t\tembed_2 = self.embed_name_dict.get(embed_2)\n\n\t\tmat1 = get_matrix(embed_1, vocab)\n\t\tmat2 = get_matrix(embed_2, vocab)\n\t\tmean_mat = np.mean([mat1, mat2], axis = 0)\n\t\tdel mat1, mat2\n\n\t\treturn mean_mat\n\n\nclass Model(nn.Module):\n\tdef __init__(self, embed_size,max_features ,maxlen ,embedding_matrix = None):\n\t\tsuper(Model, self).__init__()\n\n\t\thidden_size = 128\n\n\t\tself.embedding1 = nn.Embedding(max_features, embed_size)\n\t\tif(embedding_matrix != None):\n\t\t\tself.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))\n\t\tself.embedding1.weight.requires_grad = False\n\n\t\tself.embedding_dropout = nn.Dropout2d(0.2)\n\n\t\tself.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)\n\t\tself.gru1 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)\n\t\tself.lstm_attention1 = Attention(hidden_size*2, maxlen)\n\t\tself.gru_attention1 = Attention(hidden_size*2, maxlen)\n\n\t\tself.linear1 = nn.Linear(1536, 32)\n\t\tself.linear2 = nn.Linear(32,1)\n\t\tself.relu = nn.ReLU()\n\t\tself.tanh = nn.Tanh()\n\t\tself.dropout = nn.Dropout(0.2)\n        \n\tdef forward(self, x, _):\n\t\th_embedding1 = self.embedding1(x)\n\t\th_embedding1 = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding1, 0)))\n\n\t\th_lstm1, _ = self.lstm1(h_embedding1)\n\t\th_gru1, _ = self.gru1(h_lstm1)\n\t\th_lstm_atten1 = self.lstm_attention1(h_lstm1)\n\t\th_gru_atten1 = self.gru_attention1(h_gru1)\n\t\tavg_pool1 = torch.mean(h_gru1, 1)\n\t\tktop_pool1 = kmax_pooling(h_gru1, k = 3)\n\t\t#max_pool1, _ = torch.max(h_gru1, 1)\n\t\tconc = torch.cat((h_lstm_atten1, h_gru_atten1, avg_pool1, ktop_pool1), 1)\n\t\tconc1 = self.dropout(self.relu(self.linear1(conc)))\n\t\top = self.linear2(conc1)\n\n\t\treturn op\n\n\nclass ModelWithFeats(nn.Module):\n\tdef __init__(self, embed_size, max_features, maxlen, embedding_matrix = None):\n\t\tsuper(ModelWithFeats, self).__init__()\n\n\t\thidden_size = 128\n\n\t\tself.embedding1 = nn.Embedding(max_features, embed_size)\n\t\tif(embedding_matrix != None):\n\t\t\tself.embedding1.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))\n\t\tself.embedding1.weight.requires_grad = False\n\n\t\tself.embedding_dropout = nn.Dropout2d(0.2)\n\n\t\tself.lstm1 = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)\n\t\tself.gru1 = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)\n\t\tself.lstm_attention1 = Attention(hidden_size*2, maxlen)\n\t\tself.gru_attention1 = Attention(hidden_size*2, maxlen)\n\n\t\tself.linear1 = nn.Linear(1538, 32)\n\t\tself.linear2 = nn.Linear(32,1)\n\t\tself.relu = nn.ReLU()\n\t\tself.tanh = nn.Tanh()\n\t\tself.dropout = nn.Dropout(0.2)\n        \n\tdef forward(self, x, feats):\n\t\th_embedding1 = self.embedding1(x)\n\t\th_embedding1 = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding1, 0)))\n\n\t\th_lstm1, _ = self.lstm1(h_embedding1)\n\t\th_gru1, _ = self.gru1(h_lstm1)\n\t\th_lstm_atten1 = self.lstm_attention1(h_lstm1)\n\t\th_gru_atten1 = self.gru_attention1(h_gru1)\n\t\tavg_pool1 = torch.mean(h_gru1, 1)\n\t\tktop_pool1 = kmax_pooling(h_gru1, k = 3)\n\t\t#max_pool1, _ = torch.max(h_gru1, 1)\n\t\tconc = torch.cat((h_lstm_atten1, h_gru_atten1, avg_pool1, ktop_pool1, feats), 1)\n\t\tconc1 = self.dropout(self.relu(self.linear1(conc)))\n\t\top = self.linear2(conc1)\n\n\t\treturn op\n\n\n\n\n \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
    __stickytape_write_module('''loader.py''', '''import torch.utils.data\nimport torch\nfrom sklearn.model_selection import train_test_split, StratifiedKFold\nimport numpy as np\n\ndef GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features):\n    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n    test_local_loader = None\n\n    x_test_tensor = torch.tensor(test_X ,dtype = torch.long).to(device)\n\n    if local_test_X is not None:\n    \tx_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)\n    \ttest_feats_tensor = torch.tensor(test_feats, dtype = torch.float32).to(device)\n    \tx_test_dataset = torch.utils.data.TensorDataset(x_test_tensor, test_feats_tensor)\n\n    \tx_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)\n    \tlocal_test_feats_tensor = torch.tensor(local_test_feats, dtype = torch.float32).to(device)\n    \tprint(local_test_feats.shape, x_test_local_tensor.shape)\n    \tx_local_test_dataset = torch.utils.data.TensorDataset(x_test_local_tensor,local_test_feats_tensor)\n\n    \ttest_local_loader = torch.utils.data.DataLoader(x_local_test_dataset, batch_size = batch_size*2, shuffle = False)\n\n    elif local_test_X is None:\n    \tx_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)\n    \ttest_feats_tensor = torch.tensor(test_feats, dtype = torch.float32).to(device)\n    \tx_test_dataset = torch.utils.data.TensorDataset(x_test_tensor, test_feats_tensor)    \t\n    \n\n    test_loader = torch.utils.data.DataLoader(x_test_dataset, batch_size = batch_size*2, shuffle = False)   \n    \n\n    return test_loader, test_local_loader\n\n\ndef GetData(test_X, local_test_X, train_X ,train_Y, test_feats, local_test_feats, n_splits, batch_size ,use_extra_features= False):\n    test_loader, local_test_loader = GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features)\n    print(train_X.shape, test_X.shape)\n    if n_splits > 1:\n        splits = list(StratifiedKFold(n_splits = n_splits, shuffle=True, random_state= 165).split(train_X, train_Y))\n    else:\n        valid_index = 30000\n        splits = [[np.arange(start = 0, stop = train_X.shape[0]-valid_index), np.arange(start = train_X.shape[0] - valid_index, stop = train_X.shape[0])]]\n    train_preds = np.zeros((train_X.shape[0], ))\n    test_preds = np.zeros((test_X.shape[0], len(splits)))\n\n    local_test_preds = None\n    if(local_test_X is not None):\n    \tlocal_test_preds = np.zeros((local_test_X.shape[0], len(splits)))\n    \n    return test_loader, local_test_loader, splits, train_preds, test_preds, local_test_preds\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
    __stickytape_write_module('''trainer.py''', '''"""\nModel training methods\n"""\nimport torch.utils.data\nimport torch\nimport numpy as np\nimport copy\nfrom utils import *\nimport time\nfrom tqdm import tqdm\nfrom sklearn import metrics\n\ndef train_model(model,folds_list, test_loader, local_test_loader, n_epochs, batch_size ,validate, use_extra_features):\n    print("\\n --------training model----------")\n    optimizer = torch.optim.Adam(model.parameters())\n    \n    step_size = 300\n    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,\n                         step_size=step_size, mode='triangular2',\n                         gamma=0.99994)\n    \n    binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()\n    #l2_loss = torch.nn.MSELoss().cuda()\n    #folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]\n    train = torch.utils.data.TensorDataset(folds_list[0],folds_list[4] , folds_list[2])\n    valid = torch.utils.data.TensorDataset(folds_list[1], folds_list[5] , folds_list[3])\n    \n    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)\n    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size*2, shuffle=False)\n    \n    for epoch in range(n_epochs):\n        start_time = time.time()\n        model.train()\n        avg_loss = 0.\n        \n        for (x_full_batch_train, feat_train ,y_batch_train) in tqdm(train_loader, disable = True):\n            y_pred_train = model(x_full_batch_train, feat_train)\n            scheduler.batch_step()\n            loss = binary_cross_entropy(y_pred_train, y_batch_train)\n            optimizer.zero_grad()\n            loss.backward()\n            optimizer.step()\n            avg_loss += loss.item() / len(train_loader)\n            \n        model.eval()\n        valid_preds = np.zeros((folds_list[1].size(0)))\n        if validate == "True":\n            avg_val_loss = 0.\n            for i, (x_full_batch_val, feat_val ,y_batch_val) in enumerate(valid_loader):\n                y_pred_val = model(x_full_batch_val, feat_val).detach()\n                avg_val_loss += binary_cross_entropy(y_pred_val, y_batch_val).item() / len(valid_loader)\n                valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]\n            \n            search_result = threshold_search(folds_list[3].cpu().numpy(), valid_preds)\n            val_f1, val_threshold = search_result['f1'], search_result['threshold']\n            elapsed_time = time.time() - start_time\n            print('Epoch {}/{} \\t loss={:.4f} \\t val_loss={:.4f} \\t val_f1={:.4f} best_t={:.2f} \\t time={:.2f}s'.format(\n                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))\n        else:\n            elapsed_time = time.time() - start_time\n            print('Epoch {}/{} \\t loss={:.4f} \\t time={:.2f}s'.format(\n                epoch + 1, n_epochs, avg_loss, elapsed_time))\n    \n    model.eval()\n    valid_preds = np.zeros((folds_list[1].size(0)))\n    avg_val_loss = 0.\n    for i, (x_full_batch_val, feat_val,y_batch_val) in enumerate(valid_loader):\n        y_pred_val = model(x_full_batch_val, feat_val).detach()\n        avg_val_loss += binary_cross_entropy(y_pred_val , y_batch_val).item() / len(valid_loader)\n        valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]\n    print('Validation loss: ', avg_val_loss)\n\n    test_preds = np.zeros((len(test_loader.dataset)))\n    for i, (x_full_batch_test, feat_test) in enumerate(test_loader):\n        y_pred_test = model(x_full_batch_test, feat_test).detach()\n        test_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_test.cpu().numpy())[:, 0]\n\n    test_preds_local = np.zeros((len(local_test_loader.dataset)))\n    for i, (x_full_batch_local, feat_local) in enumerate(local_test_loader):\n        y_pred_local = model(x_full_batch_local, feat_local).detach()\n        test_preds_local[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_local.cpu().numpy())[:, 0]\n\n    return valid_preds, test_preds, test_preds_local\n\ndef trainer(splits, model_orig , train_X, train_Y, epochs, test_loader, local_test_loader ,train_preds, test_preds, local_test_preds, train_feat, batch_size,validate ,use_extra_features):\n    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n    print("\\n ---------splitting----------")\n    for split_no, (train_idx, valid_idx) in enumerate(splits):\n        \n        X_train_fold = torch.tensor(train_X[train_idx], dtype = torch.long).to(device)\n        Y_train_fold = torch.tensor(train_Y[train_idx, np.newaxis], dtype = torch.float32).to(device)\n        X_val_fold = torch.tensor(train_X[valid_idx], dtype = torch.long).to(device)\n        Y_val_fold =torch.tensor(train_Y[valid_idx, np.newaxis], dtype = torch.float32).to(device)\n        \n        \n        train_feat_fold = torch.tensor(train_feat[train_idx], dtype = torch.float32).to(device)\n        valid_feat_fold = torch.tensor(train_feat[valid_idx], dtype = torch.float32).to(device)\n        \n        folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]\n        \n        model = copy.deepcopy(model_orig)\n        model.to(device)\n        print("Split {}/{}".format(split_no+1,len(splits)))\n        pred_val_fold, pred_test_fold, pred_local_test_fold = train_model(model, folds_list,\n                                                                          test_loader, local_test_loader,\n                                                                           epochs , batch_size, validate, use_extra_features)\n        \n        train_preds[valid_idx] = pred_val_fold\n        test_preds[:, split_no] = pred_test_fold\n        local_test_preds[:, split_no] = pred_local_test_fold\n    return train_preds, test_preds, local_test_preds\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n''')
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
    	parser.add_argument("--embed_type")
    	parser.add_argument("--do_preprocess")
    	parser.add_argument("--use_extra_features")
    	parser.add_argument("--validate", default = True)
    	parser.add_argument("--local_validation")
    	parser.add_argument("--lower_text")
    	parser.add_argument("--max_vocab_words", type=int, default=90000)
    	parser.add_argument("--max_seq_len", type = int)
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
    	args = parser.parse_args()
    
    	# Load data
    	train = pd.read_csv(args.data_dir+ "/train.csv")[:10000]
    	test = pd.read_csv(args.data_dir + "/test.csv")[:100]
    	n_test = len(test) * 3
    	print(train.shape)
    
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
    	tok = TokenizerBase(train["question_text"], 
    		test["question_text"], X_local_test["question_text"], args.lower_text,
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    