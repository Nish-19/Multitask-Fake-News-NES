import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('..\\FNC_Dataset\\train_fnc_processed.csv')
test_df = pd.read_csv('..\\FNC_Dataset\\competition_test_fnc_processed.csv')

quora_df_test = pd.read_csv('..\\novelty_module\\test_fnc_quora_plain.csv')
quora_df_train = pd.read_csv('..\\novelty_module\\train_fnc_quora_plain.csv')

quora_df_test.rename(columns = {'0':'Quora_Labels'}, inplace=True)
quora_df_train.rename(columns = {'0':'Quora_Labels'}, inplace=True)

with open('FNC_Quora_Characteresticsi.txt', 'w') as infile:
	co_mat_test = coocurance_matrix(train_df.Stance, quora_df_train.Quora_Labels)
	print('##############Train FNC-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	co_mat_test = coocurance_matrix(test_df.Stance, quora_df_test.Quora_Labels)
	print('\n\n##############Test FNC-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
