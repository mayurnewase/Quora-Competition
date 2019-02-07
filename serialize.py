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
parser.add_argument("--local_validation")
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
parser.add_argument("--result_dir", type = str, default="./")	
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
	"\n\nimport argparse\n\n"
	"parser = argparse.ArgumentParser() \n"
	"parser.add_argument(\"--data_dir\", default = " + "\""+args.data_dir+"/\"" + ")\n"
	"parser.add_argument(\"--embed_dir\", default =" +"\"" +str(args.embed_dir)+ "/\"" +")\n"
	"parser.add_argument(\"--do_preprocess\", default =" + str(args.do_preprocess)+ ")\n"
	"parser.add_argument(\"--use_extra_features\", default =" + str(args.use_extra_features)+ ")\n"
	"parser.add_argument(\"--validate\", default =" + str(args.validate)+ ")\n"
	"parser.add_argument(\"--local_validation\", default ="+ str( args.lower_text)+ ")\n"
	"parser.add_argument(\"--lower_text\", default =" + str(args.lower_text)+ ")\n"
	"parser.add_argument(\"--max_vocab_words\", type=int, default="+ str(args.max_vocab_words)+ ")\n"
	"parser.add_argument(\"--max_seq_len\", type =int, default="+str(args.max_seq_len) +")\n"
	"parser.add_argument(\"--filters\", default =" + "\"" + str(args.filters) + "\"" + ")\n"
	"parser.add_argument(\"--use_embeddings\", default =" + str(args.use_embeddings)+ ")\n"
	"parser.add_argument(\"--load_glove\", default ="+ str(args.load_glove)+ ")\n"
	"parser.add_argument(\"--load_fast\", default ="+ str(args.load_fast)+ ")\n"
	"parser.add_argument(\"--load_para\", default =" + str(args.load_para)+ ")\n"
	"parser.add_argument(\"--single_matrix\", default =" + str(args.single_matrix)+ ")\n"
	"parser.add_argument(\"--mean_matrix\", default =" + str(args.mean_matrix)+ ")\n"
	"parser.add_argument(\"--concat_matrix\", default =" + str(args.concat_matrix)+ ")\n"
	"parser.add_argument(\"--splits\", type = int , default =" + str(args.splits)+ ")\n"
	"parser.add_argument(\"--epochs\", type = int , default =" + str(args.epochs)+ ")\n"
	"parser.add_argument(\"--batch_size\", type = int , default =" + str(args.batch_size)+ ")\n"
	"parser.add_argument(\"--save_result\", type = str, default="+ str(args.save_result) + ")\n"	
	"parser.add_argument(\"--result_dir\", type = str, default=" + "\""+str(args.result_dir) + "/\"" + ")\n"
	"args = parser.parse_args()"
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












































