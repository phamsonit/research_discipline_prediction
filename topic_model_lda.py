import gensim
# from diversity_measures.diversity import *
from recommendation import calculate_diversity_vector
from topic_diversity import create_topic_distribution_vector
from ultil import calculate_avg_acc


def find_optimal_k_topic(final_abstract, doc_term_matrix, dictionary, k1, k2):
    """
    find best k-topic from a range(k1,k2)
    :param final_abstract:
    :param doc_term_matrix:
    :param dictionary:
    :param k1:
    :param k2:
    :return:
    """
    # coherence = []
    max_coherence = 0
    k_optimal = 0
    for k in range(k1, k2):
        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix, num_topics=k, id2word=dictionary, passes=40,
                       iterations=200, chunksize=10000, eval_every=None, alpha='auto')

        cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=final_abstract,
                                                         dictionary=dictionary, coherence='c_v')
        print('k=' + str(k) + ", coherence=" + str(cm.get_coherence()))
        if max_coherence < cm.get_coherence():
            max_coherence = cm.get_coherence()
            k_optimal = k
        # coherence.append((k, cm.get_coherence()))
    return k_optimal  # value of k where coherence is max


def train_lda(doc_term_matrix, dictionary, k_optimal):
    Lda = gensim.models.ldamodel.LdaModel
    lda_model = Lda(doc_term_matrix, num_topics=k_optimal, id2word=dictionary, passes=40,
                   iterations=200,  chunksize=10000, eval_every=None, random_state=0, alpha='auto')
    return lda_model


def lda_get_doc_topic_matrix(lda_model, doc_term_matrix, k_topic):
    """
    create doc-topic matrix, where each row is a list of probability distribution of topics
    :param lda_model: lda model
    :param doc_term_matrix: 2d array
    :param k_topic: int - number of topics
    :return: 2d array
    """
    doc_top_matrix = []
    for i in range(len(doc_term_matrix)):
        topics = lda_model.get_document_topics(doc_term_matrix[i])
        doc_top_matrix.append(create_topic_distribution_vector(topics, k_topic))
    return doc_top_matrix


def lda_test_classification(test_data, lda_model, distance_matrix, map_top_dis):
    """
    test classification system
    :param test_data: a dataframe consists of at least two column 'abstracts':list of terms and 'disciplines': list of labels
    :param lda_model: trained lda model
    :param distance_matrix: distance matrix
    :param map_top_dis
    :return: average accuracy
    """
    nb_samples = test_data.shape[0]
    nb_topics = len(distance_matrix[0])
    dictionary = lda_model.id2word
    test_corpus = [dictionary.doc2bow(text) for text in test_data.abstracts]
    test_labels = test_data.disciplines
    test_cor_labels = list(zip(test_corpus, test_labels))
    sum_avg_acc = 0
    for doc, labels in test_cor_labels:
        predicted_topics = lda_model[doc]  # get topic probability distribution for a document
        # sort the topics based on probability distributions
        predicted_topics.sort(key=lambda y: y[1], reverse=True)
        # print(predicted_topics)
        # calculate avg accuracy based on top 3-topic probability distribution
        predicted_topic_nums = [x[0] for x in predicted_topics[:3]]
        sum_avg_acc += calculate_avg_acc(predicted_topic_nums, labels, map_top_dis)
        # create distribution vector
        dis_vector = create_topic_distribution_vector(predicted_topics, nb_topics)
        # calculate topic diversity
        dt = calculate_diversity_vector(dis_vector, distance_matrix)
        # print(dt, predicted_topics[:3])
    print("average acc: ", str(sum_avg_acc/test_data.shape[0]))
    return sum_avg_acc/nb_samples


def lda_extract_topics_words(topics_words_temp):
    """
    extract words in each topics
    :param topics_words_temp: list of tuples (int, string)
    :return: list of strings
    """
    topics_words = []
    for t_w in topics_words_temp:
        words = [x[0] for x in t_w[1]]
        topics_words.append(words)
    return topics_words


# def run_lda(df, model_path):
#     # LDA only
#     print("2: preprocessing data")
#     df_pre = data_preprocessing(df)
#     print("#after preprocessing:", df_pre.shape)
#     df_pre.to_excel(model_path + "\\input_projects.xlsx")
#
#     # split data into train and test
#     train_data, test_data = train_test_split(df_pre, test_size=0.20, random_state=42)
#     print("train data: ", train_data.shape)
#     print("test data: ", test_data.shape)
#
#     # create BoW for training model
#     dictionary = corpora.Dictionary(train_data.abstracts)
#     doc_term_matrix = [dictionary.doc2bow(doc) for doc in train_data.abstracts]
#     print("vocabulary size:", len(dictionary))
#
#     # find optimal k topic here, for example k_optimal = 10
#     nb_topics = 42  # find_optimal_k_topic(train_data.abstracts, doc_term_matrix, dictionary, 30, 50)
#     print("k topic:", nb_topics)
#
#     lda_model = train_lda(model_path+"\\_model", doc_term_matrix, dictionary, nb_topics)
#
#     print("4: calculate discipline distance matrix based on topic-term matrix")
#     topic_term_matrix = lda_model.get_topics()
#     distance_matrix = create_distance_matrix(topic_term_matrix)
#     np.savetxt(model_path + "\\distance_matrix.txt", distance_matrix, fmt="%s")
#
#     print("5: calculate discipline co-occurrence matrix based on doc-topic matrix")
#     doc_top_matrix = lda_get_doc_topic_matrix(lda_model, doc_term_matrix, nb_topics)
#     np.savetxt(model_path + "\\doc_topic_matrix.txt", doc_top_matrix, fmt="%s")
#
#     print("5.1: create co-occurrence matrix and full-network-file for VOSviewer software")
#     occ_matrix = create_occ_matrix(doc_top_matrix)
#     np.savetxt(model_path + "\\occ_matrix.txt", occ_matrix, fmt="%s")
#     create_VOSViewer_network_full(model_path + "\\occ_matrix.txt")
#
#     print("6: calculate topic diversity of all projects")
#     train_data["Diversity"] = calculate_topic_diversity(doc_top_matrix, distance_matrix)
#     print("6.1. classify projects by topic diversity scores")
#     df_IDR_classify = classify_IDR_projects(train_data)
#     df_IDR_classify.to_excel(model_path+"\\result.xlsx")
#
#     print("mapping topics to disciplines")
#     # find top-50 terms for each topic
#     topics_words = lda_extract_topics_words(lda_model.show_topics(nb_topics, num_words=50, formatted=False))
#
#     # compute mapping of topics to disciplines dictionary
#     disciplines_keywords = load_disciplines_keywords("discipline_keywords.txt")
#     map_top_dis = mapping_topics_disciplines(topics_words, disciplines_keywords)
#
#     print("test classification performance")
#     lda_test_classification(test_data, lda_model, distance_matrix, map_top_dis)




