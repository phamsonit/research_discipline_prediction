import numpy as np

from recommendation import calculate_diversity_vector
from topic_diversity import create_topic_distribution_vector
from ultil import calculate_avg_acc


def top2vec_get_doc_topic_matrix(doc_ids, model):
    """
    :param doc_ids:
    :return:
    """
    nb_topics = model.get_num_topics()
    doc_top_matrix = []
    # code to calcualte diversity of 1 document. use for loop to calculate diversity of multiple documents
    for i in doc_ids:
        doc_i = doc_ids[i]
        # get similarity scores of document i with all topics
        topic_nums, topic_score, topics_words, word_scores = model.get_documents_topics([doc_i], num_topics=nb_topics)
        # create vector of topic probability distribution
        top_dis_vector = create_dis_vector(nb_topics, topic_nums[0], topic_score[0])
        # print(top_dis_vector)
        doc_top_matrix.append(top_dis_vector)

    return doc_top_matrix


def create_dis_vector(nb_topics, topic_nums, topic_score):
    # scale the similarity scores such that sum of scores = 1
    scale_topic_score = topic_score / np.sum(topic_score)
    # print(scale_topic_score)
    # create a list of tuples of topic ids and topic score
    top_dis = list(zip(topic_nums, scale_topic_score))
    # create topic distribution vector for Rao-Stirling diversity measurement
    top_dis_vector = create_topic_distribution_vector(top_dis, nb_topics)
    return top_dis_vector


def top2vec_test_classification(test_data, model, distance_matrix, map_top_dis):
    avg_accuracy = 0
    abstracts = test_data.abstract.to_list()
    true_labels = test_data.disciplines.to_list()
    # get topics of test data
    for i in range(len(abstracts)):
        # search 3 most relevant topics of the abstracts[i]
        topics_words, word_scores, topic_scores, topic_nums = model.query_topics(abstracts[i], 3)
        # calculate avg accuracy
        acc = calculate_avg_acc(topic_nums, true_labels[i], map_top_dis)
        avg_accuracy += acc

        # calculate topic diversity
        top_dis_vector = create_dis_vector(model.get_num_topics(), topic_nums, topic_scores)
        # calculate topic diversity
        dt = calculate_diversity_vector(top_dis_vector, distance_matrix)
        # print(dt, list(zip(topic_nums, topic_scores)))

    print("avg accuracy", avg_accuracy/len(abstracts))


# def run_top2vec(df, model_path):
#     # don't need to do data preprocessing step!
#
#     # split data into train and test
#     train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)
#     print("train data: ", train_data.shape)
#     print("test data: ", test_data.shape)
#
#     print("training top2vec")
#     model = Top2Vec(train_data.abstract.to_list(), embedding_model='universal-sentence-encoder')
#     # save model to file
#     model.save(model_path+"\\top2vec_model")
#     # get topic-term matrix
#     topic_term_matrix = model.topic_vectors
#
#     # get doc-topic matrix
#     doc_ids = np.arange(train_data.shape[0])
#     doc_top_matrix = top2vec_get_doc_topic_matrix(doc_ids, model)
#
#     print("calculate distance matrix")
#     distance_matrix = create_distance_matrix(topic_term_matrix)
#
#     print("calculate topic diversity")
#     train_data["Diversity"] = calculate_topic_diversity(doc_top_matrix, distance_matrix)
#
#     print("classify projects to 4 groups, based on topic diversity scores")
#     df_IDR_classify = classify_IDR_projects(train_data)
#     df_IDR_classify.to_excel(model_path+"\\result.xlsx")
#
#     print("get mapping of topics to disciplines")
#     disciplines_keywords = load_disciplines_keywords("discipline_keywords.txt")
#     topics_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())
#     map_top_dis = mapping_topics_disciplines(topics_words, disciplines_keywords)
#
#     print("predict disciplines of projects in test data")
#     top2vec_test_classification(test_data, model, distance_matrix, map_top_dis)

