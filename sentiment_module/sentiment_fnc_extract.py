import pandas as pd
from encoder import Model

sentiment_model = Model()

data_df = pd.read_csv('/sda/rina_1921cs13/Nischal/multitask/datasets/test_fnc_agdgonly.csv') # all the feedback text was placed in a pandas data frame

# Premise will be the same
samples_pre = list(data_df['Headline']) 
text_features_pre = sentiment_model.transform(samples_pre)
sentiment_scores_pre = text_features_pre[:, 2388]
sentiment_scores_hyp_normal = []
sentiment_scores_hyp_count = []

# Sentiment score for hypothesis (document form)
sample_hyp = list(data_df['Body'])
for i, text in enumerate(sample_hyp):
	text_lst = text.split('.')
	text_features_hyp = sentiment_model.transform(text_lst)
	sentiment_scores_hyp_txt = text_features_hyp[:, 2388]
	print(sentiment_scores_hyp_txt)
	overall_score = 0
	pos_count = 0
	neg_count = 0
	for score in sentiment_scores_hyp_txt:
		overall_score+=score
		if score > 0:
			pos_count+=1
		elif score < 0:
			neg_count+=1
	overall_score = score/len(sentiment_scores_hyp_txt)
	#print('Overall Normalized Score is', overall_score)
	sentiment_scores_hyp_normal.append(overall_score)
	if pos_count >= neg_count:
		sentiment_scores_hyp_count.append(1)
	elif neg_count > pos_count:
		sentiment_scores_hyp_count.append(0)

data_df['sentiment_scores_pre'] = sentiment_scores_pre
data_df['sentiment_scores_hyp_normal'] = sentiment_scores_hyp_normal
data_df['sentiment_scores_hyp_count'] = sentiment_scores_hyp_count

data_df.to_csv('test_fnc_sentiment_scores.csv')