import pandas as pd
import csv 
from collections import Counter 
train_df = pd.read_csv('../ByteDance_Dataset/train.csv')
test_df = pd.read_csv('../ByteDance_Dataset/test_merged.csv')
tr_lst = []
ts_lst = []
tr_lst_pre = []
tr_lst_hyp = []
ts_lst_hyp = []
ts_lst_pre = []
print('Train Distribution', Counter(train_df['bd_label']))
print('Test Distribution', Counter(test_df['bd_label']))
tr_actr=0
tr_dctr = 0
for i, row in train_df.iterrows():
	if row['bd_label'] == 'unrelated':
		continue
	if row['bd_label'] == 'agreed':
		tr_actr+=1
		tr_lst.append([row['id'], row['title1_en'], row['title2_en'], row['bd_label']])
		tr_lst_pre.append([row['title1_en'], 1, row['id']])
		tr_lst_hyp.append([row['title2_en'], 1, row['id']])
	elif row['bd_label'] == 'disagreed':
		tr_dctr+=1
		tr_lst.append([row['id'], row['title1_en'], row['title2_en'], row['bd_label']])
		tr_lst_pre.append([row['title1_en'], 1, row['id']])
		tr_lst_hyp.append([row['title2_en'], 0, row['id']])
print("Train agreed", tr_actr)
print("Train disagreed", tr_dctr)

ts_actr=0
ts_dctr = 0
for i, row in test_df.iterrows():
	if row['bd_label'] == 'unrelated':
		continue
	if row['bd_label'] == 'agreed':
		ts_actr+=1
		ts_lst.append([row['id'], row['title1_en'], row['title2_en'], row['bd_label']])
		ts_lst_pre.append([row['title1_en'], 1, row['id']])
		ts_lst_hyp.append([row['title2_en'], 1, row['id']])
	elif row['bd_label'] == 'disagreed':
		ts_dctr+=1
		ts_lst.append([row['id'], row['title1_en'], row['title2_en'], row['bd_label']])
		ts_lst_pre.append([row['title1_en'], 1, row['id']])
		ts_lst_hyp.append([row['title2_en'], 0, row['id']])
print("Test agreed", ts_actr)
print("Test disagreed", ts_dctr)

# Convert to files accordingly
with open('data/train_ag_dg_only.csv', 'w', newline= '', encoding='utf-8') as outfile:
	csv_writer = csv.writer(outfile, delimiter=',')
	for obj in tr_lst:
		csv_writer.writerow(obj)

with open('data/train_ag_dg_premise.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in tr_lst_pre:
		tsv_writer.writerow(obj)

with open('data/train_ag_dg_hyp.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in tr_lst_hyp:
		tsv_writer.writerow(obj)

# Convert Test to files accordingly
with open('data/test_ag_dg_only.csv', 'w', newline= '', encoding='utf-8') as outfile:
	csv_writer = csv.writer(outfile, delimiter=',')
	for obj in ts_lst:
		csv_writer.writerow(obj)

with open('data/test_ag_dg_premise.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in ts_lst_pre:
		tsv_writer.writerow(obj)

with open('data/test_ag_dg_hyp.tsv', 'w', newline = '', encoding='utf-8') as outfile:
	tsv_writer = csv.writer(outfile, delimiter='\t')
	for obj in ts_lst_hyp:
		tsv_writer.writerow(obj)
