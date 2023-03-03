"""
A metadata based framework for discipline prediction and interdisciplinarity calculation
author: Hoang Son Pham, ECOOM-Hasselt University
email: hoangson.pham@uhasselt.be
"""
import ast
import os
import pickle
import numpy as np
import pandas as pd
from gensim import corpora
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from top2vec import Top2Vec


import topic_model_llda_implementation as labeled_lda
from data_preprocessing import label_encoding
from recommendation import recommend_disciplines, calculate_predicted_discipline_diversity
from text_processing import text_preprocessing
from topic_diversity import create_distance_matrix
from topic_model_lda import train_lda, lda_get_doc_topic_matrix, lda_extract_topics_words
from topic_model_llda import predict_discipline_proba, load_llda_model
from topic_model_top2vec import top2vec_get_doc_topic_matrix
from ultil import load_distance_matrix, load_discipline_list


def run_llda_model(df_sub, output_path):
    # df_sub['disciplines'] = df_sub['disciplines'].apply(lambda x: literal_eval(x))
    labeled_documents = list(zip(df_sub.abstracts.to_list(), df_sub.disciplines.to_list()))
    # create llda model
    llda_model = labeled_lda.LldaModel(labeled_documents=labeled_documents, alpha_vector=None, eta_vector=None,
                                   common_topic=False, theta_star=None)

    # training until convergent or number of iteration == 10
    while True and llda_model.iteration <= 10:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.is_convergent(method="beta", delta=0.01):
            break

    # save the model to file
    llda_model.save_model_to_dir(output_path)

    return llda_model.beta, llda_model.theta, llda_model.topics


def run_lda_model(train_data):
    # preprocessing data
    train_data = text_preprocessing(train_data)
    # create dictionary
    dictionary = corpora.Dictionary(train_data.abstracts)
    # create feature matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_data.abstracts]
    # find optimal k topic here, for example k_optimal = 10
    nb_topics = 42  # find_optimal_k_topic(train_data.abstracts, doc_term_matrix, dictionary, 30, 50)
    print("k topic:", nb_topics)
    # train lda model
    model = train_lda(doc_term_matrix, dictionary, nb_topics)
    # save model
    model.save(output_model_path + database_name + '_lda_model')
    # get topics terms matrix
    topic_term_matrix = model.get_topics()
    # get documents topics matrix
    doc_top_matrix = lda_get_doc_topic_matrix(model, doc_term_matrix, nb_topics)
    # get the most (50) relevant words for each topic
    topics_words = lda_extract_topics_words(model.show_topics(nb_topics, num_words=50, formatted=False))
    return topic_term_matrix, doc_top_matrix, nb_topics


def run_top2vec_model(train_data):
    print("train top2vec")
    model = Top2Vec(train_data.abstracts.to_list(), embedding_model='universal-sentence-encoder')
    num_topics = model.get_num_topics()
    print("k topic:", num_topics)
    # save model to file
    model.save(output_model_path + database_name + "_top2vec_model")
    # get topic-term matrix
    topic_term_matrix = model.topic_vectors
    # get doc-topic matrix
    doc_ids = np.arange(train_data.shape[0])
    doc_top_matrix = top2vec_get_doc_topic_matrix(doc_ids, model)
    return topic_term_matrix, doc_top_matrix, num_topics


def train_classifier(X, y, model_name):
    if model_name == 'kNN':
        train_model = KNeighborsClassifier().fit(X, y)
    else:
        if model_name == 'RF':
            train_model = MultiOutputClassifier(RandomForestClassifier()).fit(X, y)
        else:
            if model_name == 'BG':
                train_model = MultiOutputClassifier(GradientBoostingClassifier(max_features='auto')).fit(X, y)
            else:
                train_model = None
    return train_model


if __name__ == "__main__":
    task = 'test'
    # select database name
    database_name = 'FRIS'
    # database_name = 'Dimensions'

    # select topic model
    topic_model_name = 'top2vec'

    # define output paths
    output_data_path = 'outputs\\data\\'
    output_model_path = 'outputs\\models\\'

    input_data_path = output_data_path + database_name + '_project_data.csv'
    distance_matrix_path = output_data_path + database_name+'_distance_matrix.txt'
    topic_probability_path = output_data_path + database_name + '_topic_probability.txt'
    discipline_list_path = output_data_path + database_name + '_discipline_list.txt'

    t2v_model_path = output_model_path + database_name + "_top2vec_model"
    llda_model_path = output_model_path + database_name + "_llda_model"
    clas_model_path = output_model_path + database_name+'_classification_model.sav'

    # Note:  Data extraction and data pre-processing are done before
    print('====> read project data')
    # project data prepared by the pre-processing step
    df_train = pd.read_csv(input_data_path)
    # df_train, df_test = train_test_split(df, test_size=0.2)
    if task == 'train':
        # covert disciplines in form of text to list
        df_train['disciplines'] = df_train['disciplines'].apply(lambda x: ast.literal_eval(x))
        # df_train = df_train_temp.reset_index()
        print('====> encoding labels')
        # create binary_label, discipline_list and int_label
        int_label = label_encoding(df_train, database_name, output_data_path)

        print('====> run top2vec to discover topic probability')
        top2vec_topic_term_matrix, top2vec_doc_top_matrix, top2vec_num_topics = run_top2vec_model(df_train)
        # save topic probability distribution to disk
        np.savetxt(topic_probability_path, top2vec_doc_top_matrix, fmt="%s")

        print("===> run llda to find discipline probability distribution")
        # create a directory to store llda model
        if not os.path.exists(llda_model_path):
            # if the directory is not present then create it.
            os.makedirs(llda_model_path)

        # train llda model
        llda_topic_term_probability, llda_doc_top_probability, llda_topics_name = run_llda_model(df_train, llda_model_path)

        # create distance matrix
        distance_matrix = create_distance_matrix(llda_topic_term_probability)
        # save distance matrix to disk
        np.savetxt(distance_matrix_path, distance_matrix, fmt='%s')

        # train ML classifier
        X_train = top2vec_doc_top_matrix
        y_train = int_label
        # train classifier
        # RF: Random Forest
        # BG: Gradient Boosting
        # kNN: k nearest neighbors
        cl_model = train_classifier(X_train, y_train, 'RF')
        # save classifier model to disk
        # save the model to disk
        pickle.dump(cl_model, open(clas_model_path, 'wb'))
    else:
        # get a test sample data
        sample = df_train.abstracts.values[5]
        # load Top2vec model
        t2v_model = Top2Vec.load(t2v_model_path)

        # search topic probability distribution over the sample
        topics_words, word_scores, topic_scores, topic_nums = t2v_model.query_topics(sample, t2v_model.get_num_topics())

        # sort the order of topics and scores
        l_tuples = list(zip(topic_scores, topic_nums))
        l_tuples.sort(key=lambda y: y[1])
        # get topic probability distribution
        sample_topic_proba = [x[0] for x in l_tuples]

        # load the classification model from disk
        ml_model = pickle.load(open(clas_model_path, 'rb'))
        # predict label of the sample data
        y_pred = ml_model.predict([sample_topic_proba])

        # load distance matrix
        distance_matrix = load_distance_matrix(distance_matrix_path)
        # distance_matrix = 1 - distance_matrix
        # load Dimensions discipline list
        discipline_list = load_discipline_list(discipline_list_path)

        # find discipline probability by L-LDA
        # load llda model
        llda_model = load_llda_model(llda_model_path)
        d_t = llda_model.theta

        # dis_proba = predict_discipline_proba(sample, llda_model, discipline_list)
        # print(dis_proba)

        # recommend close disciplines
        rd_filter = recommend_disciplines(y_pred[0], distance_matrix, discipline_list, True)

        # calculate diversity of recommended disciplines
        rs = calculate_predicted_discipline_diversity(rd_filter, distance_matrix, discipline_list)
        print('# IDR score:', round(rs, 2))

    pass



