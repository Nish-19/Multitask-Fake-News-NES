import pandas as pd

train_df = pd.read_csv('../ag_dg_only/train_fnc_agdgonly.csv')
test_df = pd.read_csv('../ag_dg_only/test_fnc_agdgonly.csv')

train_em_df = pd.read_csv('fnc_goemotion.csv')
test_em_df = pd.read_csv('fnc_goemotion_test.csv')

femotion_tr = []
femotion_ts = []

for i in range(len(train_df)):
	em1 = train_df.loc[i, 'Emotion_Labels']
	em2 = train_em_df.loc[i, 'femotion']
	label = train_df.loc[i, 'Stance']
	if label == 'agree':
		if em1 == 1 or em2 == 1:
			femotion_tr.append(1)
		else:
			femotion_tr.append(0)
	if label == 'disagree':
		if em1 == 0 or em2 == 0:
			femotion_tr.append(0)
		else:
			femotion_tr.append(1)

for i in range(len(test_df)):
	em1 = test_df.loc[i, 'Emotion_Labels']
	em2 = test_em_df.loc[i, 'femotion']
	label = test_df.loc[i, 'Stance']
	if label == 'agree':
		if em1 == 1 or em2 == 1:
			femotion_ts.append(1)
		else:
			femotion_ts.append(0)
	if label == 'disagree':
		if em1 == 0 or em2 == 0:
			femotion_ts.append(0)
		else:
			femotion_ts.append(1)

train_em_df["com_femotion"] = femotion_tr
train_em_df.to_csv('com_femotion_tr.csv', index = False)

test_em_df["com_femotion"] = femotion_ts
test_em_df.to_csv('com_femotion_ts.csv', index = False)
