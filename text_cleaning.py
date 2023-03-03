import spacy
import nltk
import re


nlp = spacy.load('en_core_web_sm')
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}


def clean_all(df, col_name):
    # encode for only ascii characters
    df[col_name] = df[col_name].map(ascii_rm)

    # lowercase texts
    df[col_name] = df[col_name].map(lambda x: x.lower())

    # lemmatize words
    df[col_name] = df[col_name].astype(str).map(lemma)

    # remove punctuation
    df[col_name] = df[col_name].map(punc_n)

    # filter only english comments/non blank comments
    df = df[df[col_name] != ""]
    df['language'] = df[col_name].map(get_language)
    df = df.loc[df['language'] == 'english']
    df = df.drop('language', axis=1)

    return df


def ascii_rm(text):
    try:
        text = text.encode('ascii', errors='ignore')
        return text
    except:
        print(text)


def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key=lambda x: x[1])[0]


def punc_n(text):
    regex = re.compile('[' + re.escape('!"#%&\'()*+,-./:;<=>?@[\\]^_`{|}~') + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    nopunct_words = nopunct.split(' ')
    filter_words = [word.strip() for word in nopunct_words if word != '']
    words = ' '.join(filter_words)
    return words


def lemma(text):
    lemmatized = nlp(text)
    lemmatized_final = ' '.join([word.lemma_ for word in lemmatized if word.lemma_ != '\'s'])
    return lemmatized_final
