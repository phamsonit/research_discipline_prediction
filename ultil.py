import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import itertools


def kmean_cluster_topics(top_term_matrix, nb_cluster):
    """
    cluster topics based on word distributions
    :param top_term_matrix: topic term matrix
    :param nb_cluster: number of clusters
    :return:
    """
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(top_term_matrix)
    return kmeans.labels_


def plot_top_words(components_, feature_names, n_top_words, top_frequent_topics_idx, title):
    num_subplots = len(top_frequent_topics_idx)
    cols = 4
    rows = num_subplots // cols
    fig, axes = plt.subplots(2, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    ax_count = 0
    for topic_idx, topic in enumerate(components_):
        if topic_idx in top_frequent_topics_idx:
            top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            ax = axes[ax_count]
            ax_count += 1
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 15})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=15)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def calculate_UMass(wi, wj, doc_term_matrix, feature_names):
    """
    calculate umass score for a pair of words wi, wj
    :param wi: string - i_th word
    :param wj: string - j_th word
    :param doc_term_matrix: 2-dimension array
    :param feature_names: list of string - feature names
    :return: float number - umass score
    """
    try:
        # count co-occurrences of wi and wj in documents
        d_wi_wj = 0
        wi_id = feature_names.index(wi)
        wj_id = feature_names.index(wj)
        for d in range(len(doc_term_matrix)):
            if doc_term_matrix[d][wi_id] != 0 and doc_term_matrix[d][wj_id] != 0:
                d_wi_wj += 1
        # count occurrences of w1
        d_wi = len([i for i in doc_term_matrix[:, wi_id] if i > 0])
        umass = math.log((d_wi_wj + 1) / d_wi)
        return umass
    except Exception as e:
        print(e, wi, wj)
        return 0


def calculate_model_topic_coherence(W, doc_term_matrix, feature_names):
    """
    :param W: list of list terms
    :param doc_term_matrix: 2-dimension array - doc term matrix
    :param feature_names: list of strings - list of terms
    :return:
    """
    topic_umass = []
    for w in W:
        sum_umass = 0
        pair_count = 0
        for wi in range(len(w) - 1):
            for wj in range(wi + 1, len(w)):
                sum_umass += calculate_UMass(w[wi], w[wj], doc_term_matrix, feature_names)
                pair_count += 1
        topic_umass.append(sum_umass/pair_count)
    return topic_umass


def calculate_model_topic_diversity(W):
    """
    Dieng et al. (2020) proposes a topic diversity metric that considers the percentage of unique words in the top
    25 words of all topics (a value close to 1 indicates more varied topics).
    :param W: 2-dimension list of string - list of keywords
    :return: list of float numbers - topic diversities (percentage of unique keywords)
    """
    # for each list of keywords count unique words
    topic_diversity = []
    for w in W:
        # collect all words except words in w
        all_words = create_word_corpus(W, w)
        # count unique keywords in w
        count = 0
        for keyword in w:
            if keyword not in all_words:
                count += 1
        # calculate percentage of unique keywords and add to the result list
        topic_diversity.append(count / len(w))

    return topic_diversity


def create_word_corpus(W, w):
    """
    collect all unique words in W except words in w
    :param W: 2-dimension list of string - list of keywords
    :param w: list of string - a list of keywords, i.e., a row in W
    :return: set of string - a set of unique keywords
    """
    temp = W.copy()
    temp.remove(w)
    return set(itertools.chain(*temp))


def create_binary_doc_term_matrix(W, terms):
    """
    create doc term binary matrix
    :param W: input corpus where W[m][n] = id of term in terms
    :param terms: list of term
    :return: binary matrix
    """
    m = []
    for doc in W:
        t = [0] * len(terms)
        for term_id in doc:
            t[term_id] = 1
        m.append(t)
    return m


def classify_IDR_projects(df):
    """
    implementation of CSS algorithm to divide projects into 4 groups based on Diversity score
    :param df: a dataframe
    :return: a dataframe with a IDR_levels column
    """
    # assign IDR level to projects
    df_result = pd.DataFrame(columns=df.columns)
    df_result['IDR_levels'] = []
    for i in range(1, 4):
        # calculate global average
        gl_average = df['Diversity'].sum() / df.shape[0]
        # select projects that have average score less than gl_average score
        df_i = df[df['Diversity'] < gl_average].copy()
        # assign current IDR level to the selected projects
        df_i['IDR_levels'] = i  # np.array(np.full(shape=df_i.shape[0], fill_value=i, dtype=np.int))
        # append df_i to result
        df_result = df_result.append(df_i, ignore_index=True)
        # update df by removing projects that have average < gl_average
        df = df[df['Diversity'] >= gl_average].copy()
    # the rest of projects are assigned to level 4
    df_i = df.copy()
    df_i['IDR_levels'] = 4  # np.array(np.full(shape=df_i.shape[0], fill_value=4, dtype=np.int))
    # append df_i to result
    df_result = df_result.append(df_i, ignore_index=True)
    return df_result


def create_occ_matrix(doc_topic_matrix, threshold):
    """
    create discipline co-occurrence matrix from output of topic model, i.e, doc-topic matrix
    :param doc_topic_matrix: a 2-dimension array
    :param threshold
    :return: a 2-dimension array
    """
    # load llda model
    nb_discipline = len(doc_topic_matrix[0])
    occ_matrix = np.zeros((nb_discipline, nb_discipline))
    # get project-discipline matrix
    nb_row = len(doc_topic_matrix)
    nb_col = len(doc_topic_matrix[0])
    for i in range(nb_col):
        for j in range(nb_col):
            for k in range(nb_row):
                if doc_topic_matrix[k][i] >= threshold and doc_topic_matrix[k][j] >= threshold:  # TODO:get average value of two columns
                    occ_matrix[i][j] += 1
                    occ_matrix[j][i] += 1
    return occ_matrix


def calculate_avg_acc(predicted_topics, true_disciplines, map_top_dis):
    # calculate accuracy based on topic_nums and true discipline
    # print("predicted_topic_nums", predicted_topics)
    predicted_discipline_codes = convert_to_predicted_disciplines(predicted_topics, map_top_dis)
    # print("predicted_discipline_codes", predicted_discipline_codes)
    # print("true_disciplines_codes", true_disciplines)
    true_prediction_count = 0
    for true_discipline in true_disciplines:
        if true_discipline in predicted_discipline_codes:
            true_prediction_count += 1
    avg_acc = true_prediction_count/len(true_disciplines)
    return avg_acc


def convert_to_predicted_disciplines(predicted_topic_nums, map_top_dis):
    """
    convert discipline ids to corresponding discipline codes
    :param predicted_topic_nums: list of int
    :param map_top_dis: dictionary with key is topic id, value is discipline code
    :return:
    """
    # TODO change map_top_dis to map_dis_top
    predicted_disciplines = set()
    for t in predicted_topic_nums:
        # find this topic number in map_dis_top
        # for each item in the map
        for item in map_top_dis.items():
            # get the list of topic ids
            map_top_nums = item[1]
            # if predicted topic t is in the list of topic ids
            if t in map_top_nums:
                # add discipline code to the predicted disciplines
                predicted_disciplines.add(item[0])
    return predicted_disciplines


def load_discipline_list(file_name):
    d_list = []
    open_file = open(file_name, 'r')
    data = open_file.readlines()
    for d in data:
        d_list.append(d.strip())
    return d_list


def load_distance_matrix(file_name):
    dm = np.loadtxt(file_name)
    return dm