# Have to change the path accordingly
import pandas as pd
import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
dct_1 = unpickle('test_fnc_quora_emb_4ag_5dg_4.pickle')
dct_2 = unpickle('test_fnc_quora_emb_4ag_5dg_8.pickle')
np_1 = list(dct_1.values())
np_2 = list(dct_2.values())  
df_1 = pd.read_csv('test_fnc_4ag_5dg_4.csv')
df_2 = pd.read_csv('test_fnc_4ag_5dg_8.csv')
#test_df = pd.read_csv('competition_test_fnc_processed.csv')
test_df = pd.read_csv('FNC_Dataset/competition_test_fnc_processed.csv')
lab_lst = []
final_np = []
assert len(df_1) == len(df_2) == len(test_df)
for i in range(len(test_df)):
	if test_df.loc[i, 'Stance'] == 'agree':
		if df_2.loc[i, 'Novelty_Quora'] == 1:
			lab_lst.append(1)
			final_np.append(np_2[i])
		elif df_1.loc[i, 'Novelty_Quora'] == 1:
			lab_lst.append(1)
			final_np.append(np_1[i])
		else:
			lab_lst.append(0)
			final_np.append(np_1[i])
	elif test_df.loc[i, 'Stance'] == 'disagree':
		if df_1.loc[i, 'Novelty_Quora'] == 0:
			lab_lst.append(0)
			final_np.append(np_1[i])
		elif df_2.loc[i, 'Novelty_Quora'] == 0:
			lab_lst.append(0)
			final_np.append(np_2[i])
		else:
			lab_lst.append(1)
			final_np.append(np_1[i])
	else:
		lab_lst.append(df_1.loc[i, 'Novelty_Quora'])
		final_np.append(np_1[i])
new_df = pd.DataFrame(columns = ['Novelty_Quora'])
new_df['Novelty_Quora'] = lab_lst
new_df.to_csv('test_4ag_5dg_4_8_combined.csv')
fnc_nv_4ag_5dg_4_8_combine = np.stack(final_np)
print('Shape of array is', fnc_nv_4ag_5dg_4_8_combine.shape)
np.save(file = 'fnc_nv_4ag_5dg_4_8_combine_test.npy', arr=fnc_nv_4ag_5dg_4_8_combine)
