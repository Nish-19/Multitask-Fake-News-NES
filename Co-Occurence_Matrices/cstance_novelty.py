import pandas as pd

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('cstance_train.csv')
test_df = pd.read_csv('cstance_test_new.csv')

quora_df_train = pd.read_csv('train_costance_nv_all8.csv')
quora_df_test = pd.read_csv('test_costance_nv_all8.csv')

with open('Costance_Quora_Characterestics_all8_dup.txt', 'w') as infile:
	co_mat_test = coocurance_matrix(train_df.stance, quora_df_train.Novelty_Quora)
	print('##############Train CS-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	co_mat_test = coocurance_matrix(test_df.stance, quora_df_test.Novelty_Quora)
	print('\n\n##############Test CS-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
