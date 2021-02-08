# Extracting agreed and disagreed into a file
import pandas as pd
import os 
file = "../FNC_Dataset/train_fnc_mt2nd.csv"
ip_df = pd.read_csv(file)
lst = []
# Note - 0 is the dummy label
agree_count=0
disagree_count=0
for i, row in ip_df.iterrows():
	if row['Stance'] == 'agree' and agree_count<400:
		agree_count+=1
		lst.append([row['Body'].encode('utf-8'), row['Headline'].encode('utf-8'), 1])
	elif row['Stance'] == 'disagree' and disagree_count<500:
		disagree_count+=1
		lst.append([row['Body'].encode('utf-8'), row['Headline'].encode('utf-8'), 0])
# Writing to a txt file
with open(file.split('.')[0]+"_quora_convert_4ag_5dg.txt", 'w', newline = '') as outfile:
	for obj in lst:
		outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')

# For appending the new bd dataset with the train of quora
# Python program to 
# demonstrate merging 
# of two files 
  
data = data2 = "" 
  
# Reading data from file1 
with open('data/quora/train.txt', encoding='utf-8') as fp: 
    data = fp.read()
  
# Reading data from file2 
with open('train_fnc_processed_quora_convert_4ag_5dg.txt', encoding='utf-8') as fp: 
    data2 = fp.read()
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('combined_train_fnc_4ag_5dg.txt', 'w', encoding='utf-8') as fp: 
    fp.write(data) 