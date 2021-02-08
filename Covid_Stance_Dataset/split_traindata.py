import pandas as pd 
train_df = pd.read_csv('cstance_train.csv')
total_length = len(train_df)
ph1len = int(total_length * 0.2)
first_dataset = train_df.iloc[:ph1len, :]
second_dataset = train_df.iloc[ph1len:, :]
first_dataset.to_csv('train_cs_mt1st.csv', index = False)
second_dataset.to_csv('train_cs_mt2nd.csv', index = False)
print('Shape of the 1st Dataset is', first_dataset.shape)
print('Shape of the 2nd Dataset is', second_dataset.shape)