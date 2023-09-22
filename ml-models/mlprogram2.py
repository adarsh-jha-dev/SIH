import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from collections import Counter
from keras.layers import LSTM
from nltk.corpus import stopwords #removing stepwords 
from nltk.tokenize import word_tokenize # tokenize the sentence
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
keras.layers.Dropout
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras.layers import Embedding, Dense, GlobalAveragePooling1D # deep learning techniques
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)
data=pd.read_csv("dataset.csv")
data=data.drop_duplicates()
import re
from nltk.corpus import stopwords
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[^a-zA-Z0-9\s]+', '', text)
    text = re.sub('\w*\d\w*', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    text = re.sub('\s+', ' ', text).strip()
    return text
data['sentence'] = data['sentence'].apply(clean_text)
import nltk
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)
data['sentence'] = data['sentence'].apply(stem_text)
max_words = 10000 
max_len = 200 
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(data['sentence']) 
sequences_train = tokenizer.texts_to_sequences(data['sentence']) 
word_index = tokenizer.word_index 
data_train = pad_sequences(sequences_train, maxlen = max_len)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['sentence'])
y = data['is_seed']
train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2) 
x_train_res, y_train_res =sm.fit_resample(x_train, y_train.ravel()) 
print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(x_train_res, y_train_res)
print("Test accuracy of random forest classifier: (%.2f%%)"%(rf_model.score(x_test,y_test)*100))
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import xgboost
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model = xgboost.XGBClassifier()
model.fit(x_train_res,y_train_res)
y_pred=model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy of XGB: %.2f%%" % (accuracy * 100.0))
kfold = KFold(n_splits=10)
results= cross_val_score(model,x_val, y_val, cv=kfold)
print("Validation Accuracy of XGB: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))