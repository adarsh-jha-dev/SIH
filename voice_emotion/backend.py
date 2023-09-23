
import pandas as pd
import numpy as np
import pickle
df = pd.read_csv("./emotion-dataset.csv")
df['Emotion'].value_counts()
import neattext.functions as nfx
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size= 0.3, random_state=42)
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(X_train,y_train)
pickle.dump(pipe_lr,open("texttoemotion.pkl","wb"))

