import re
import pymorphy2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

from collections import Counter
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from razdel import tokenize
from wordcloud import WordCloud


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Выбор признака для передачи в pipeline
    """
    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x[[self.column]]


def add_stop_words(dataframe: pd.DataFrame,
                   k_words: int) -> list:
    """
    Получить список стоп-слов, которые наиболее часто встречаются в документе
    :param dataframe:
    :param k_words: кол-во наиболее часто повторяющихся уникальных слов
    :return:
    """

    split_words = dataframe['text'].values
    split_words = " ".join(split_words)
    split_words = split_words.split()
    _counter = Counter(split_words).most_common(k_words)
    n_words = [i[0] for i in _counter]

    return list(set(n_words))


class Cleaner(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    @staticmethod
    def clean_text(text):
        """
        Очистка текста от сложноинтерпритируемых символов
        :param text: Текст для очистки
        :return: Очищенные слова из поданного текста, разделенные " "
        """
        if not isinstance(text, str):
            text = str(text)

        text = text.lower()
        text = text.strip('\n').strip('\r').strip('\t')
        text = re.sub("-", ' ', str(text))
        text = re.sub("-\s\r\n\|-\s\r\n|\r\n", '', str(text))
        text = re.sub("[0-9]|[-—.,:;_%©«»?*!@#№$^•·&()]|[+=]|[[]|[]]|[/]|", '', text)
        text = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', text)
        text = re.sub(r'[\xad]|[\s+]', ' ', text.strip())
        text = re.sub("\n", ' ', text)
        stopwords_ru = stopwords.words('russian')
        tokens = list(tokenize(text))
        words = [_.text for _ in tokens]
        words = [w for w in words if w not in stopwords_ru]

        return " ".join(words)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x[self.column] = x[self.column].apply(lambda q: self.clean_text(q), 1)
        return x[[self.column]]


class Lemmatizer(BaseEstimator, TransformerMixin):

    def __init__(self, column, words_to_add):
        self.column = column
        self.cache = {}
        self.morph = pymorphy2.MorphAnalyzer()
        self.stopword_ru = stopwords.words('russian')
        self.words_to_add = words_to_add

    def lemmatization(self, text):
        """
        Приведение словоформы к нормальной форме.
        В отличие от стемминга так сохраним больше смысловой нагрузки
        :param text: Текст для лемматизации
        :return: Словарные формы поданных слов
        """
        if not isinstance(text, str):
            text = str(text)

        tokens = list(tokenize(text))
        words = [_.text for _ in tokens]

        words_lem = []
        for w in words:
            if w[0] == '-':
                w = w[1:]
            if len(w) > 1:
                if w in self.cache:
                    words_lem.append(self.cache[w])
                else:
                    temp_cache = self.cache[w] = self.morph.parse(w)[0].normal_form
                    words_lem.append(temp_cache)

        stopwords_ru = list(set(self.stopword_ru + self.words_to_add))
        words_lem_without_stopwords = [i for i in words_lem if i not in stopwords_ru]

        return ' '.join(list(words_lem_without_stopwords))

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x[self.column] = x[self.column].apply(lambda q: self.lemmatization(q), 1)
        return x[self.column].values.astype('U')


def get_corpus(data):
    """
    Получение списка всех слов в корпусе
    :param data: Данные
    :return: список слов в корпусе
    """
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus


def str_corpus(corpus):
    """
    Получение текстовой строки из списка слов
    :param corpus: список слов
    :return: строка из списка слов в корпусе
    """
    _corpus = ''
    for i in corpus:
        _corpus += ' ' + i
    _corpus = _corpus.strip()
    return _corpus


def get_cloud(corpus):
    """
    Получение облака слов
    :param corpus: корпус слов из метода get_corpus
    :return: Облако слов
    """
    stopwords_ru = stopwords.words('russian')
    word_cloud = WordCloud(background_color='white',
                           stopwords=stopwords_ru,
                           width=3000,
                           height=2500,
                           max_words=200,
                           random_state=42
                           ).generate(str_corpus(corpus))
    return word_cloud


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Возвращает plot матрицы ошибок
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return plt.show()
