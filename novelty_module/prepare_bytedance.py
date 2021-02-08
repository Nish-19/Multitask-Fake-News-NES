# Extracting agreed and disagreed into a file
import pandas as pd
import os 
file = "../ByteDance_Dataset/train_bd_mt2nd.csv"
ip_df = pd.read_csv(file)
lst = []
# Note - 0 is the dummy label
for i, row in ip_df.iterrows():
	if row['bd_label'] == 'agreed':
		lst.append([row['title1_en'].encode('utf-8'), row['title2_en'].encode('utf-8'), 1])
	elif row['bd_label'] == 'disagreed':
		lst.append([row['title1_en'].encode('utf-8'), row['title2_en'].encode('utf-8'), 0])
# Writing to a txt file
with open(file.split('.')[0]+"_bd_quora_convert.txt", 'w') as outfile:
	for obj in lst:
		outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')

data = data2 = "" 
  
# Reading data from file1 
with open('data/quora/train.txt', encoding='utf-8') as fp: 
    data = fp.read()
  
# Reading data from file2 
with open('train_bd_quora_convert.txt', encoding='utf-8') as fp: 
    data2 = fp.read()
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('combined_train_bd.txt', 'w', encoding='utf-8') as fp: 
    fp.write(data) 