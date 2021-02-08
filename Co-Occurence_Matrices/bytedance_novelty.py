import pandas as pd

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('..\\ByteDance_Dataset\\train.csv')
test_df = pd.read_csv('..\\ByteDance_Dataset\\test_merged.csv')

quora_df_test = pd.read_csv('..\\novelty_module\\test_quora_predictions.csv')
quora_df_train = pd.read_csv('..\\novelty_module\\train_quora_predictions.csv')

quora_df_test.rename(columns = {'0':'Quora_Labels'}, inplace=True)
quora_df_train.rename(columns = {'0':'Quora_Labels'}, inplace=True)

with open('BD_Novelty_Quora_Characterestics-New.txt', 'w') as infile:
	co_mat_train = coocurance_matrix(train_df.bd_label, quora_df_train.Quora_Labels)
	print('##############Train BD-Novelty-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_train, file=infile)
	co_mat_test = coocurance_matrix(test_df.bd_label, quora_df_test.Quora_Labels)
	print('\n\n##############Test BD-Novelty-Quora-BD Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)

# sns.heatmap(co_mat_train)
# sns.heatmap(co_mat_test)
# plt.show()