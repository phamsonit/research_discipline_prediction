import numpy as np
from scipy.spatial import distance

# from diversity_measures.diversity import calculate_diversity_vector
from recommendation import calculate_diversity_vector


def create_distance_matrix(topic_term_matrix):
    """
    create distance matrix from topic-term matrix
    :param topic_term_matrix:
    :return: distance matrix
    """
    row_numbers = len(topic_term_matrix)
    distance_matrix = np.zeros((row_numbers, row_numbers))
    for i in range(row_numbers):
        for j in range(row_numbers):
            # Euclidean distance
            # distance_i_j = distance.euclidean(topic_term_matrix[i], topic_term_matrix[j])
            # Manhattan distance
            # distance_i_j = distance.cityblock(topic_term_matrix[i], topic_term_matrix[j])
            # cosine distance
            distance_i_j = distance.cosine(topic_term_matrix[i], topic_term_matrix[j])
            # add distance to distance matrix
            distance_matrix[i][j] = round(distance_i_j, 4)
            distance_matrix[j][i] = round(distance_i_j, 4)
    return distance_matrix


def calculate_topic_diversity(pro_top_matrix, distance_matrix):
    """
    calculate topic diversity of multiple projects
    :param pro_top_matrix:
    :param distance_matrix:
    :return:
    """
    result = []
    for pro_i in range(len(pro_top_matrix)):
        topic_distributions = pro_top_matrix[pro_i]
        rs = calculate_diversity_vector(topic_distributions, distance_matrix)
        result.append(rs)
    return result


def create_topic_distribution_vector(dis_topics, k_topic):
    """
    create probability distribution vector with length = k_topic
    such that the order of topics must be the same as they are in the distance matrix
    :param dis_topics: a list of tuples. each tuple includes a topic id and probability distribution
    :param k_topic: number of topics
    :return: a list of probability distribution of topics
    """
    dis_vector = np.zeros(k_topic)
    for t in dis_topics:
        top_id = t[0]
        dis_vector[top_id] = t[1]
    return dis_vector
