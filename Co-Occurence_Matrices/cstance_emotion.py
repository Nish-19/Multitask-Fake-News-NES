import pandas as pd

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('cstance_train.csv')
test_df = pd.read_csv('cstance_test_new.csv')

emotion_df_train = pd.read_csv('cstance_train_em.tsv_k_bal_numb_predictions_bin.csv')
emotion_df_test = pd.read_csv('cstance_test_new_em.tsv_k_bal_numb_predictions_bin.csv')

with open('CS_BIN_BAL_Emotion_Characterestics.txt', 'w') as infile:
	print('\n\n##############Train-CS Co-Occurance Matrix##############\n', file=infile)
	co_mat_test = coocurance_matrix(train_df.stance, emotion_df_train.Emotion_Label)
	print(co_mat_test, file=infile)
	print('\n\n##############Test-CS Co-Occurance Matrix##############\n', file=infile)
	co_mat_test = coocurance_matrix(test_df.stance, emotion_df_test.Emotion_Label)
	print(co_mat_test, file=infile)