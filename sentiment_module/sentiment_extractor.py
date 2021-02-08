import pandas as pd
from encoder import Model

sentiment_model = Model()

data_df = pd.read_csv('/sda/rina_1921cs13/Nischal/multitask/datasets/cstance_test_new.csv') # all the feedback text was placed in a pandas data frame

samples_pre = list(data_df['premise']) 
sample_hyp = list(data_df['text'])

text_features_pre = sentiment_model.transform(samples_pre)
text_features_hyp = sentiment_model.transform(sample_hyp)

sentiment_scores_pre = text_features_pre[:, 2388]
sentiment_scores_hyp = text_features_hyp[:, 2388]

data_df['sentiment_scores_pre'] = sentiment_scores_pre
data_df['sentiment_scores_hyp'] = sentiment_scores_hyp

data_df.to_csv('test_cs_sentiment_scores.csv')