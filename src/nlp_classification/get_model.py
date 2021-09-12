import os
import nltk
import pandas as pd
import pickle

from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from environment_reference import EnvironmentReference
from fill_functions import FeatureSelector, Cleaner, Lemmatizer
from fill_functions import add_stop_words

from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

load_dotenv()

data = pd.read_csv(os.getenv(EnvironmentReference.DATA_PATH))
save_path = os.getenv(EnvironmentReference.SAVE_MODEL_PATH)

X, y = data.drop('label', 1), data['label']

data_stopwords = add_stop_words(data, 20)

print('Making pipeline...')

classifier = Pipeline([('selector', FeatureSelector(column='text')),
                       ('cleaner', Cleaner(column='text')),
                       ('lemma', Lemmatizer(column='text',
                                            words_to_add=data_stopwords)),
                       ('tfidf', TfidfVectorizer(sublinear_tf=True,
                                                 strip_accents='unicode',
                                                 analyzer='word',
                                                 token_pattern=r'\w{1,}',
                                                 ngram_range=(1, 1),
                                                 max_features=10000)),
                       ('clf', MultinomialNB(alpha=0.5, fit_prior=True))])

print('Learning model...')

classifier.fit(X, y)

print('Saving model...')

# Сериализуем пайплайн
with open(f'{save_path}MNB_model.pkl', 'wb') as pkl:
    pickle.dump(classifier, pkl)

print('Done')
