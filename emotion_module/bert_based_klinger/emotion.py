import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.stats import spearmanr
from transformers import *
import random
random.seed(1)
import csv,cv2, string
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from keras.activations import softmax
from nltk.corpus import stopwords
import os,re, codecs
import keras, keras_metrics
from tqdm import tqdm
from keras import objectives
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import *
from gensim.models import KeyedVectors
from tensorflow.keras.optimizers import SGD,Adam
from keras.utils import np_utils,to_categorical
from imutils import paths
from keras.layers.merge import Concatenate, Average, concatenate
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import asarray,zeros
import seaborn as sns
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')
from transformers import TFBertModel
#Lines to run the code on GPU
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


np.set_printoptions(suppress=True)
print(tf.__version__)

tokenizer = BertTokenizer.from_pretrained('/sda/rina_1921cs13/tf-hub-cache/1/assets/vocab.txt')

df_data = pd.read_csv('/sda/rina_1921cs13/multitask/emo_embed/Klinger.csv')
#################################Preprocess##########################

#Remove Punctuation#########################
def rem_punc(text):
	no_punc = "".join([c for c in text if c not in string.punctuation])
	return no_punc

df_data['text'] = df_data['text'].apply(lambda x:rem_punc(x))

print(df_data['text'].head())
print("punctuation removed")
"""
#########################Tokenization##############

tokenizer = RegexpTokenizer(r'\w+')

df_data['text'] = df_data['text'].apply(lambda x:tokenizer.tokenize(x.lower()))

print(df_data['text'].head())
print("text tokenized")

#########################Stopwords removal###########
def remove_stopwords(text):
	words = [w for w in text if w not in stopwords.words('english')]
	return words

df_data['text'] = df_data['text'].apply(lambda x:remove_stopwords(x))

print(df_data['text'].head())
print("stopwords removed")
#####################Lemitization######################

lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
	lem_text = [lemmatizer.lemmatize(i) for i in text]
	return lem_text

df_data['text']= df_data['text'].apply(lambda x:word_lemmatizer(x))

print(df_data['text'].head())
print("text lemmatization")
"""
print('train shape =', df_data.shape)
train, test = train_test_split(df_data, test_size=0.2)
# calculate the maximum document length

output_categories = list(df_data.columns[2:])
input_categories = list(df_data.columns[[1]])

print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)

def _convert_to_transformer_inputs(text, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids, input_masks, input_segments = return_id(
        text , None, 'longest_first', max_sequence_length)
    
    return [input_ids, input_masks, input_segments]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t = instance.text
        #print(t)

        ids, masks, segments = \
        _convert_to_transformer_inputs(t, tokenizer, max_sequence_length)
        
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.relu(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

#----------------------Finding labels------------------------------------

train['target'] = train.label.astype('category').cat.codes
train['num_words'] = train.text.apply(lambda x : len(x.split()))


test['target'] = test.label.astype('category').cat.codes
test['num_words'] = test.text.apply(lambda x : len(x.split()))


y_train = train['label']
y_test =test['label']

#-----------------------------------------KFOLD--------------------------------
actual=[]
target_names = ['joy', 'anger', 'sadness', 'disgust', 'fear', 'surprise', 'trust', 'anticipation']
MAX_SEQUENCE_LENGTH = max(train['num_words'])
print(MAX_SEQUENCE_LENGTH)
#------------------model architecture------------------------------
checkpoint = ModelCheckpoint(filepath='/sda/rina_1921cs13/multitask/Rina/Emotion/emo_weight2.h5',monitor='val_acc', mode='max', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=20)
############################################Visual feature extraction#################################

def create_model():
    t_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    t_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    t_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    bert_model = TFBertModel.from_pretrained('/sda/rina_1921cs13/march/bert-base-uncased-tf_model.h5', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    t_embedding = bert_model(t_id, attention_mask=t_mask, token_type_ids=t_atn)[0]
    q = tf.keras.layers.GlobalAveragePooling1D()(t_embedding)
    model = tf.keras.models.Model(inputs=[t_id, t_mask, t_atn], outputs=q)
    
    return model

outputs = compute_output_arrays(train, output_categories)
inputs = compute_input_arrays(train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

test_inputs = compute_input_arrays(test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
def on_epoch_end(self):
        pass

def reduce_sum(x):
  return K.sum(x, axis=-1)
########################################Textual Feature Extraction#####################################
K.clear_session()
model = create_model()
train_text_features = model.predict(inputs)
test_text_features = model.predict(test_inputs)

print(train_text_features.shape)
print(test_text_features.shape)


##########################################Joint Model###################################################
#------------------------------------------Text Branch--------------------------------------------------
text_input = Input(shape=(train_text_features.shape[1],))
x = tf.keras.layers.Dense(768, activation='relu')(text_input)
#x = tf.keras.layers.Reshape((8, 198))(x)
#x = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True))(x)
#x = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True))(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
#x = tf.keras.layers.Dropout(0.4)(x1)
#x = tf.keras.layers.Dense(64, activation='relu')(x2)
x = tf.keras.layers.Reshape((16, 16))(x)
x = attention()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
z  = tf.keras.layers.Dense(32, activation='relu',name='emo')(x)
z = tf.keras.layers.Dense(8, activation='softmax')(z)

class_model = tf.keras.models.Model(inputs=text_input, outputs=z)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
class_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
class_model.summary()
history = class_model.fit(x=[train_text_features], y=[to_categorical(np.array(y_train))], validation_split=0.2, epochs=100,callbacks=[es,checkpoint], batch_size=64,shuffle=True,verbose=1)

model_new = tf.keras.models.Model(inputs=text_input, outputs=z)
model_new.load_weights('/sda/rina_1921cs13/multitask/Rina/Emotion/emo_weight2.h5', by_name=True)

model_new1 = Model(inputs= text_input, outputs=class_model.get_layer('emo').output)
model_new1.load_weights('/sda/rina_1921cs13/multitask/Rina/Emotion/emo_weight2.h5', by_name=True)

#test_preds1= model_new1.predict([test_text_features])

test_preds= model_new.predict([test_text_features])
rounded = [round(x[0]) for x in test_preds]
out3 = np.array(rounded, dtype='int64')
#out3 = np.argmax(test_preds, axis=1)

"""
actual=[]
res1=[]

for i in out3:
	if i==0:
		res1.append("fake")
		print("fake")
	else:
		res1.append("real")
		print("real")

test_accuracy=accuracy_score(y_test, out3)


for i in y_test:
	if i==0:
		actual.append("fake")
	else:
		actual.append("real")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=10)
pca_result1 = pca.fit_transform(test_preds1)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

#Run T-SNE on the PCA features.
tsne = TSNE(n_components=3, verbose = 1, n_iter=1000)
tsne_results = tsne.fit_transform(pca_result1)

fig = plt.figure(figsize=(8,8))
y_test_cat = np_utils.to_categorical(y_test, num_classes = 2)
color_map = np.argmax(y_test_cat, axis=1)

for cl in range(2):
    indices = np.where(color_map==cl)
    indices = indices[0]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
plt.title('joint feature representation')
plt.legend(['Fake','Real'], loc='upper left')
fig.savefig('/sda/rina_1921cs13/Journal/All_Result/ti_cnn/tsne11.png')

"""
#--------------Classification report computation---------------------------------
class_rep=classification_report(y_test, out3, target_names=target_names)
#print(test_accuracy)
#print("\n")
print("Classification report",class_rep)
print("\n")


print("Confusion matrix",confusion_matrix(y_test, out3))
print("\n")

"""
precision = precision_score(y_test, out3)
recall = recall_score(y_test, out3)
f1 = f1_score(y_test, out3)


#Writting all the metrics in a file
with open("/sda/rina_1921cs13/Journal/All_Result/ti_cnn/Result11.txt",'a') as f:
	f.write( 'Results ' + '\n' )
	f.write( 'Test Accuracy = ' + str(test_accuracy*100) + '\n' )
	f.write( 'Test Precision = ' + str(precision*100) + '\n' )
	f.write( 'Test Recall = ' + str(recall*100)+ '\n' )
	f.write( 'Test F_score = ' + str(f1*100)+ '\n' )
	f.write( 'Test F1_Score = ' + str(((2*precision*recall)/(precision+recall))*100) + '\n' )

	f.write("Actual"+"\t"+"res"+"\n")
	for i,j in zip(actual,res1):
		f.write(str(i)+"\t"+str(j)+"\n")
f.close()

import matplotlib as mpl
mpl.use('Agg')
fig1 = plt.figure() 
plt.plot(history.history['acc'],label = 'acc', color ='b')
plt.plot(history.history['val_acc'],label = 'val_acc', color ='r')
plt.title('model training accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'], loc='upper left')

fig2 = plt.figure()
plt.plot(history.history['loss'],label = 'loss', color ='b')
plt.plot(history.history['val_loss'],label = 'val_loss', color ='g')
plt.title('model training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss']) 


fig1.savefig('/sda/rina_1921cs13/Journal/All_Result/ti_cnn/acc11.png')
fig2.savefig('/sda/rina_1921cs13/Journal/All_Result/ti_cnn/loss11.png')
"""
