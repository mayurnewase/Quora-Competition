"""
make 1 script
copy all file in main.py
accept params and hardcode them in argparser default
accept line numbers to clean file imports
"""

import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
parser.add_argument("--embed_dir")
parser.add_argument("--do_preprocess")
parser.add_argument("--use_extra_features")
parser.add_argument("--validate", default = True)
parser.add_argument("--local_validation", default = True)
parser.add_argument("--lower_text")
parser.add_argument("--max_vocab_words", type=int, default=90000)
parser.add_argument("--max_seq_len", type = int, default=72)
parser.add_argument("--filters", default="")
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
parser.add_argument("--files_to_copy", type = str)
parser.add_argument("--file_impo_lines", type=str)
parser.add_argument("--main_parser_lines", type=str)
args = parser.parse_args()


w = open("singletone.py", mode = "w")
for file in args.files_to_copy.split(","):
	r = open(file, "r")
	data =r.read()
	w.write("#-------- COPYING FILE " + str(file) + "--------------\n\n")
	w.write(data)
	r.close()

w.write(
	"\n"
	"class ArgsDummy:\n"+
	"\tdata_dir ="+ "\"" + str(args.data_dir)+ "/\""+"\n"
	"\tembed_dir ="+ "\"" + str(args.embed_dir)+ "/\""+"\n"
	"\tdo_preprocess ="+ "\""+ str(args.do_preprocess)+ "\""+"\n"
	"\tuse_extra_features ="+ "\""+ str(args.use_extra_features)+ "\""+"\n"
	"\tvalidate ="+ "\""+ str(args.validate)+ "\""+"\n"
	"\tlocal_validation ="+ "\""+ str(args.local_validation)+ "\""+"\n"
	"\tlower_text ="+ "\""+ str(args.lower_text)+ "\""+"\n"
	"\tmax_vocab_words ="+ str(args.max_vocab_words)+"\n"
	"\tmax_seq_len =" +str(args.max_seq_len)+"\n"
	"\tfilters ="+ "\"" +str(args.filters)+ "\""+"\n"
	"\tuse_embeddings ="+ "\"" +str(args.use_embeddings)+ "\""+"\n"
	"\tload_glove ="+ "\"" +str(args.load_glove)+ "\""+"\n"
	"\tload_fast ="+ "\"" +str(args.load_fast)+ "\""+"\n"
	"\tload_para ="+ "\"" +str(args.load_para)+ "\""+"\n"
	"\tsingle_matrix ="+ "\"" +str(args.single_matrix)+ "\""+"\n"
	"\tmean_matrix =" + "\""+str(args.mean_matrix)+ "\""+"\n"
	"\tconcat_matrix ="+ "\"" +str(args.concat_matrix)+ "\""+"\n"
	"\tsplits =" +str(args.splits)+"\n"
	"\tepochs =" +str(args.epochs)+"\n"
	"\tbatch_size =" +str(args.batch_size)+"\n"
	"\tsave_result =" +str(args.save_result)+"\n"
	"\tresult_dir =" + "\""+str(args.result_dir)+ "/\""+"\n"
	"args = ArgsDummy()\n\n"
	)	

r = open("main.py", "r")
#data = r.read()
w.write("\n\n#-------- COPYING MAIN + --------------\n\n")

file_impo = np.arange(int(args.file_impo_lines.split(",")[0]), int(args.file_impo_lines.split(",")[1])+1)
main_parse = np.arange(int(args.main_parser_lines.split(",")[0]), int(args.main_parser_lines.split(",")[1])+1)
cut_lines = list(file_impo) + list(main_parse)
print(cut_lines)
for no,line in enumerate(r):
	if no+1 not in cut_lines:
		w.write(line)
	else:
		print(line)
	

r.close()
w.close()












































