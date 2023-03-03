from stop_word_list import *
import nltk
import pandas as pd


def calculate_bigram_scores(abstracts):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents(
        [abstract.split() for abstract in abstracts])
    # Filter only those that occur at least 50 times
    finder.apply_freq_filter(50)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)
    return bigram_scores


def calculate_trigram_scores(abstracts):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_documents(
        [abstract.split() for abstract in abstracts])
    # Filter only those that occur at least 50 times
    finder.apply_freq_filter(50)
    trigram_scores = finder.score_ngrams(trigram_measures.pmi)
    return trigram_scores


# Filter for bigrams with only noun-type structures
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in stop_word_list or bigram[1] in stop_word_list:
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True


# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in stop_word_list or trigram[-1] in stop_word_list or trigram[1] in stop_word_list:
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True


def create_ngrams(abstracts):
    bigram_scores = calculate_bigram_scores(abstracts)
    trigram_scores = calculate_trigram_scores(abstracts)
    bigram_pmi = pd.DataFrame(bigram_scores)
    bigram_pmi.columns = ['bigram', 'pmi']
    bigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)
    trigram_pmi = pd.DataFrame(trigram_scores)
    trigram_pmi.columns = ['trigram', 'pmi']
    trigram_pmi.sort_values(by='pmi', axis=0, ascending=False, inplace=True)
    # Can set pmi threshold to whatever makes sense - eyeball through and select threshold
    # where n-grams stop making sense
    # choose top 5000 ngrams in this case ranked by PMI that have noun like structures
    filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram: \
                                                      bigram_filter(bigram['bigram']) \
                                                      and bigram.pmi > 5, axis=1)][:5000]
    filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: \
                                                         trigram_filter(trigram['trigram']) \
                                                         and trigram.pmi > 5, axis=1)][:5000]
    bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
    trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if
                len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
    return bigrams, trigrams


# Concatenate n-grams
def replace_ngram(abstract, bigrams, trigrams):
    for gram in trigrams:
        abstract = abstract.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        abstract = abstract.replace(gram, '_'.join(gram.split()))
    return abstract


# Filter for only nouns
def noun_only(abstract):
    pos_abstract = nltk.pos_tag(abstract)
    filtered = [word[0] for word in pos_abstract if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered


def text_preprocessing(df):
    # df columns = [..., 'title', 'keywords',  'abstract', 'discipline']
    # df['abstracts'] = df.title.str.cat(df.abstract, sep=' ')

    # clean abstracts
    # df_clean = clean_all(df, 'abstracts')

    # prepare bigrams and trigrams
    abstract_w_ngrams = pd.DataFrame(df.abstracts.copy())
    bigrams, trigrams = create_ngrams(abstract_w_ngrams.abstracts)
    abstract_w_ngrams.abstracts = abstract_w_ngrams.abstracts.map(lambda x: replace_ngram(x, bigrams, trigrams))

    # tokenize abstract + remove stop words + remove names + remove words with less than 2 characters
    abstract_w_ngrams = abstract_w_ngrams.abstracts.map(lambda x: [word for word in x.split() \
                                                                       if word not in stop_word_list \
                                                                       and word not in english_names \
                                                                       and len(word) > 2])
    # Filter for only nouns
    df.abstracts = abstract_w_ngrams.map(noun_only)

    return df



