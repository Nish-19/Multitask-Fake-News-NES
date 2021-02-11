#This code creates the dataset from Corpus.csv which is downloadable from the
#internet well known dataset which is labeled manually by hand. But for the text
#of tweets you need to fetch them with their IDs.
import tweepy

# Twitter Developer keys here
# It is CENSORED
consumer_key = 'Fill_appropriately'
consumer_key_secret = 'Fill_appropriately'
access_token = 'Fill_appropriately'
access_token_secret = 'Fill_appropriately'

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# This method creates the training set
def createTrainingSet(corpusFile, targetResultFile):
    import csv
    import time

    counter = 0
    corpus = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[0], "label": row[1]})

    sleepTime = 2
    trainingDataSet = []
    infile = open('Scrape_Statistics.txt', 'w')
    abs_file = open('Text_Absent_Ids.txt', 'w')
    pre_ctr = 0
    abs_ctr = 0
    abs_lst = []
    for i, tweet in enumerate(corpus):
        print(i)
        try:
            tweetFetched = api.get_status(tweet["tweet_id"], tweet_mode="extended")
            #print("Tweet fetched" + tweetFetched.text)
            pre_ctr+=1
            tweet["text"] = tweetFetched.full_text
            trainingDataSet.append(tweet)
            time.sleep(sleepTime)
            if i < 10:
                print(tweet["text"])

        except:
            abs_ctr+=1
            abs_lst.append(tweet['tweet_id'])
            print("Inside the exception - no:2")
            continue

    with open(targetResultFile, 'w', newline = '') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"].encode('ascii', 'ignore'), tweet["label"]])
            except Exception as e:
                print(e)
    print('Present Text:', pre_ctr, '\n', 'Absent Text:', abs_ctr, file=infile)
    print(abs_lst, file=abs_file)
    return trainingDataSet

# Code starts here
# This is corpus dataset
corpusFile = "COVID-CQ.csv"
# This is my target file
targetResultFile = "tweet_covidcq_dataset.csv"
# Call the method
resultFile = createTrainingSet(corpusFile, targetResultFile)