import pandas as pd

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('..\\FNC_Dataset\\train_ag_dg_only_fnc.csv')
test_df = pd.read_csv('..\\FNC_Dataset\\test_ag_dg_only_fnc.csv')

# emotion_df_hyp = pd.read_csv('test_merged_hypothesis.tsv_k_numb_predictions_bin.csv')
# emotion_df_pre = pd.read_csv('test_merged_premise.tsv_k_numb_predictions_bin.csv')

emotion_df_hyp = pd.read_csv('..\\emotion_module\\train_ag_dg_premise_fnc.tsv_new_k_bal_numb_predictions_bin.csv')
emotion_df_pre = pd.read_csv('..\\emotion_module\\train_ag_dg_hyp_fnc.tsv_new_k_bal_numb_predictions_bin.csv')

emotion_df_hyp.rename(columns = {'0':'Emotion_Label'}, inplace=True)
emotion_df_pre.rename(columns = {'0':'Emotion_Label'}, inplace=True)

assert len(train_df) == len(emotion_df_hyp) == len(emotion_df_pre)

with open('Train_FNC_BIN_BAL_Emotion_Characterestics.txt', 'w') as infile:
	print(len(train_df.Stance), len(emotion_df_pre.Emotion_Label))
	co_mat_test = coocurance_matrix(train_df.Stance, emotion_df_pre.Emotion_Label)
	print('\n\n##############Train-Premise-FNC Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	co_mat_test = coocurance_matrix(train_df.Stance, emotion_df_hyp.Emotion_Label)
	print('\n\n##############Train-Hypothesis-FNC Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)

relation_lst = []
for i, row in emotion_df_pre.iterrows():
	if emotion_df_pre.loc[i, 'Emotion_Label'] == 0 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 0:
		relation_lst.append('a')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 0 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 1:
		relation_lst.append('b')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 1 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 0:
		relation_lst.append('c')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 1 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 1:
		relation_lst.append('d')
	else:
		print('Some Problem')
train_df['Combined_Results'] = relation_lst
train_df.to_csv('combined_train_results_bin_bal.csv')
train_df = pd.read_csv('combined_train_results_bin_bal.csv')

with open('FNC_Train_Emotion_BIN_BAL_Combined_Cat.txt', 'w') as infile:
	co_mat_test = coocurance_matrix(train_df.Stance, train_df.Combined_Results)
	print('##############Train FNC-Emotion-Premise-Hypothesis Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)

emotion_df_hyp = pd.read_csv('test_ag_dg_premise_fnc.tsv_new_k_bal_numb_predictions_bin.csv')
emotion_df_pre = pd.read_csv('test_ag_dg_hyp_fnc.tsv_new_k_bal_numb_predictions_bin.csv')

emotion_df_hyp.rename(columns = {'0':'Emotion_Label'}, inplace=True)
emotion_df_pre.rename(columns = {'0':'Emotion_Label'}, inplace=True)

assert len(test_df) == len(emotion_df_hyp) == len(emotion_df_pre)

with open('Test_FNC_BIN_BAL_Emotion_Characterestics.txt', 'w') as infile:
	co_mat_test = coocurance_matrix(test_df.Stance, emotion_df_pre.Emotion_Label)
	print('\n\n##############Test-Premise-FNC Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	co_mat_test = coocurance_matrix(test_df.Stance, emotion_df_hyp.Emotion_Label)
	print('\n\n##############Test-Hypothesis-FNC Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)

relation_lst = []
for i, row in emotion_df_pre.iterrows():
	if emotion_df_pre.loc[i, 'Emotion_Label'] == 0 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 0:
		relation_lst.append('a')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 0 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 1:
		relation_lst.append('b')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 1 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 0:
		relation_lst.append('c')
	elif emotion_df_pre.loc[i, 'Emotion_Label'] == 1 and emotion_df_hyp.loc[i, 'Emotion_Label'] == 1:
		relation_lst.append('d')
	else:
		print('Some Problem')
test_df['Combined_Results'] = relation_lst
test_df.to_csv('combined_test_results_bin_bal.csv')
test_df = pd.read_csv('combined_test_results_bin_bal.csv')

with open('FNC_Test_Emotion_BIN_BAL_Combined_Cat.txt', 'w') as infile:
	co_mat_test = coocurance_matrix(test_df.Stance, test_df.Combined_Results)
	print('##############Test FNC-Emotion-Premise-Hypothesis Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)