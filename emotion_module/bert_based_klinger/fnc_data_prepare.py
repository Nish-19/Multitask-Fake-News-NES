import pandas as pd
import csv 
from collections import Counter 
train_df = pd.read_csv('../FNC_Dataset/train_fnc_processed.csv')
test_df = pd.read_csv('../FNC_Dataset/competition_test_fnc_processed.csv')
tr_lst = []
ts_lst = []
tr_lst_pre = []
tr_lst_hyp = []
ts_lst_hyp = []
ts_lst_pre = []
print('Train Distribution', Counter(train_df['Stance']))
print('Test Distribution', Counter(test_df['Stance']))
tr_actr=0
tr_dctr = 0
for i, row in train_df.iterrows():
	if row['Stance'] == 'unrelated':
		continue
	if row['Stance'] == 'agree':
		tr_actr+=1
		tr_lst.append([row['Body ID'], row['Body'], row['Headline'], row['Stance']])
		tr_lst_pre.append([row['Body'], 1, row['Body ID']])
		tr_lst_hyp.append([row['Headline'], 1, row['Body ID']])
	elif row['Stance'] == 'disagree':
		tr_dctr+=1
		tr_lst.append([row['Body ID'], row['Body'], row['Headline'], row['Stance']])
		tr_lst_pre.append([row['Body'], 1, row['Body ID']])
		tr_lst_hyp.append([row['Headline'], 0, row['Body ID']])
print("Train agree", tr_actr)
print("Train disagree", tr_dctr)

ts_actr=0
ts_dctr = 0
for i, row in test_df.iterrows():
	if row['Stance'] == 'unrelated':
		continue
	if row['Stance'] == 'agree':
		ts_actr+=1
		ts_lst.append([row['Body ID'], row['Body'], row['Headline'], row['Stance']])
		ts_lst_pre.append([row['Body'], 1, row['Body ID']])
		ts_lst_hyp.append([row['Headline'], 1, row['Body ID']])
	elif row['Stance'] == 'disagree':
		ts_dctr+=1
		ts_lst.append([row['Body ID'], row['Body'], row['Headline'], row['Stance']])
		ts_lst_pre.append([row['Body'], 1, row['Body ID']])
		ts_lst_hyp.append([row['Headline'], 0, row['Body ID']])
print("Test agree", ts_actr)
print("Test disagree", ts_dctr)

# Convert to files accordingly
with open('data/train_ag_dg_only_fnc.csv', 'w', newline= '', encoding='utf-8') as outfile:
	csv_writer = csv.writer(outfile, delimiter=',')
	csv_writer.writerow(['id', 'Body', 'Headline', 'Stance'])
	for obj in tr_lst:
		csv_writer.writerow(obj)

with open('data/train_ag_dg_premise_fnc.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in tr_lst_pre:
		tsv_writer.writerow(obj)

with open('data/train_ag_dg_hyp_fnc.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in tr_lst_hyp:
		tsv_writer.writerow(obj)

# Convert Test to files accordingly
with open('data/test_ag_dg_only_fnc.csv', 'w', newline= '', encoding='utf-8') as outfile:
	csv_writer = csv.writer(outfile, delimiter=',')
	csv_writer.writerow(['id', 'Body', 'Headline', 'Stance'])
	for obj in ts_lst:
		csv_writer.writerow(obj)

with open('data/test_ag_dg_premise_fnc.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in ts_lst_pre:
		tsv_writer.writerow(obj)

with open('data/test_ag_dg_hyp_fnc.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in ts_lst_hyp:
		tsv_writer.writerow(obj)
