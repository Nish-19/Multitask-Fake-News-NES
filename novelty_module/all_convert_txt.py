# Extracting agreed and disagreed into a file
import pandas as pd
import os

def create_dataset(ip_df, name):
	lst = []
	for i, row in ip_df.iterrows():
		lst.append([row['Body'].encode('utf-8'), row['Headline'].encode('utf-8'), 0])
	# Writing to a txt file
	with open(name.split('.')[0]+".txt", 'w', newline = '') as outfile:
		for obj in lst:
			outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')


file = "../FNC_Dataset/train_fnc_processed.csv"
ip_df = pd.read_csv(file)
create_dataset(ip_df, 'train_fnc_processed.csv')

file = "../FNC_Dataset/competition_test_fnc_processed.csv"
ip_df = pd.read_csv(file)
create_dataset(ip_df, 'competition_test_fnc_processed.csv')

file = "../ByteDance_Dataset/train.csv"
ip_df = pd.read_csv(file)
create_dataset(ip_df, 'train.csv')

file = "../ByteDance_Dataset/test_merged.csv"
ip_df = pd.read_csv(file)
create_dataset(ip_df, 'test_merged.csv')