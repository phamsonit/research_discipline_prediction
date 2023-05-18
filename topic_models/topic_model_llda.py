import topic_models.topic_model_llda_implementation as labeled_lda

from ultil import *


def train_llda_model(labeled_documents, alpha, eta, theta_star, common_topic, model_path):
    # new a Labeled LDA llda-model
    llda_model = labeled_lda.LldaModel(labeled_documents=labeled_documents, alpha_vector=alpha, eta_vector=eta,
                                common_topic=common_topic, theta_star=theta_star)

    # training until convergent or number of iteration == 10
    while True and llda_model.iteration < 10:
        print("iteration %s sampling..." % (llda_model.iteration + 1))
        llda_model.training(1)
        print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
        print("delta beta: %s" % llda_model.delta_beta)
        if llda_model.is_convergent(method="beta", delta=0.01):
            break
    # save the model to file
    llda_model.save_model_to_dir(model_path)
    return llda_model


def save_topic_term_matrix(llda_model, save_model_dir):
    np.savetxt(save_model_dir + '_topic_term_matrix.txt', llda_model.beta, fmt='%s')


def save_doc_topic_matrix(llda_model, save_model_dir):
    # get topics of the model
    doc_top_matrix = []
    topics_header = llda_model.topics
    # add topics' label to the first line of the matrix
    doc_top_matrix.append(topics_header)
    # add the doc-topic matrix
    doc_top_matrix.extend(llda_model.theta)
    # print("Doc-Topic Matrix: \n", llda_model.theta)
    np.savetxt(save_model_dir + '\\doc_topic_matrix.txt', doc_top_matrix, fmt='%s')


def get_top_topic_keywords(llda_model, k):
    """
    get top k keywords for every topics
    :param llda: llda model
    :param k: integer - number of keywords
    :return: list of string - list of keywords of topics
    """
    topic_keywords = []
    for t in llda_model.topics:
        top_words = llda_model.top_terms_of_topic(t, k)
        top_keyword_str = [x[0] for x in top_words]
        topic_keywords.append(top_keyword_str)
    return topic_keywords


def load_llda_model(save_model_dir):
    # load model from disk
    llda_model = labeled_lda.LldaModel()
    llda_model.load_model_from_dir(save_model_dir, load_derivative_properties=False)
    return llda_model


def update_llda_modele(llda_model):
    pass
    # # update
    # print("before updating: ", llda_model)
    # update_labeled_documents = [("new example test example test example test example test", ["example", "test"])]
    # llda_model.update(labeled_documents=update_labeled_documents)
    # print("after updating: ", llda_model)

    # train again
    # llda_model.training(iteration=100, log=True)
    # training again until convergent
    # while True:
    #     print("iteration %s sampling..." % (llda_model.iteration + 1))
    #     llda_model.training(1)
    #     print("after iteration: %s, perplexity: %s" % (llda_model.iteration, llda_model.perplexity()))
    #     print("delta beta: %s" % llda_model.delta_beta)
    #     if llda_model.is_convergent(method="beta", delta=0.01):
    #         break


def evaluate_model_performance(llda_model, model_dir):
    # create a file to write result to
    f = open(model_dir + '\\model_performance.txt', 'w')
    # print("calculate model topic diversity")
    top_topic_keywords = get_top_topic_keywords(llda_model, 25)
    topic_diversity = calculate_model_topic_diversity(top_topic_keywords)
    print("## average topic diversity: ", sum(topic_diversity) / len(topic_diversity))
    f.write("## average topic diversity: " + str(sum(topic_diversity) / len(topic_diversity)) + "\n")

    # print("calculate model topic coherence: UMASS")
    tf_matrix = create_binary_doc_term_matrix(llda_model.W, llda_model.terms)
    topic_coherence = calculate_model_topic_coherence(top_topic_keywords, np.array(tf_matrix), llda_model.terms)
    print("## average topic coherence: ", sum(topic_coherence) / len(topic_coherence))
    f.write("## average topic coherence: " + str(sum(topic_coherence) / len(topic_coherence)) + "\n")

    # print("calculate sum of each topic distribution")
    # frequent_topics = sum(llda.theta[0:])

    # print("cluster topics based on word distribution")
    # cluster_ids = ultil.kmean_cluster_topics(llda.beta, 10)

    # print top-25 keywords in each topic
    f.write("## top topic-keywords \n")
    count = 0
    for topic in llda_model.topics:
        top_words = llda_model.top_terms_of_topic(topic, 25)
        # print(topic, '\t', topic_diversity[count], '\t', topic_coherence[count], '\t', top_words)
        f.write(topic + '\t' + str(topic_diversity[count]) + '\t' + str(topic_coherence[count]) + '\t' + str(
            top_words) + "\n")
        count += 1


# def predict_project_disciplines(model_path, test_documents, test_labels):
#     """
#     predict probability distributions of topics in input unlabelled documents and calculate topic diversity
#     :param test_documents: list of unlabelled documents
#     :param model_path: path to llda model
#     :return: None
#     """
#     test_data = list(zip(test_documents, test_labels))
#     # load distance matrix
#     distance_matrix = np.loadtxt(model_path+"\\distance_matrix.txt")
#     # load llda model
#     llda_model = load_llda_model(model_path)
#     # get list of topics
#     topics = llda_model.topics
#     # inference topics of documents
#     # note: the result of topics may be different for different training, because gibbs sampling is a random algorithm
#     top_file = open(model_path + '\\discipline_prediction.txt', 'w')
#     avg_acc = 0
#     for doc, true_labels in test_data:
#         topic_distributions = llda_model.inference(document=doc, iteration=100, times=10)
#         # calculate average classification accuracy based on top-10 topic distributions
#         avg_acc += calculate_classification_accuracy(topic_distributions[:10], true_labels)
#         # calculate topic diversity of the project
#         # short topic distribution based on topic ids
#         # the order of topic ids must be the same as they are in the distance matrix
#         t = []
#         for i in topics:
#             for j in topic_distributions:
#                 if i == j[0]:
#                     t.append(j[1])
#         rs = calculate_Rao_Stirling_diversity(t, distance_matrix)
#
#         # print(rs, '\t', topic_distributions)
#         # TODO print code and name of predicted discipline
#         top_file.write(str(rs)+'\t'+str(topic_distributions)+"\n")
#     print("average accuracy = ", avg_acc/len(test_documents))
#     top_file.write("average accuracy = " + str(avg_acc/len(test_documents)))
#     top_file.flush()
#     top_file.close()

def predict_discipline_proba(doc, llda_model, dis_list):
    topic_distributions = llda_model.inference(document=doc, iteration=100, times=10)
    result = []
    for i in topic_distributions[:2]:
        code = i[0]
        for j in dis_list:
            if code in j:
                result.append(j)
    return result


def create_llda_topic_prob_vector(topic_prob, topics):
    t = []
    for i in topics:
        for j in topic_prob:
            if i == j[0]:
                t.append(j[1])
    return t

#
# def run_llda(df, model_path, alpha, eta, theta_star, common_topic):
#
#     print("2: preprocessing data")
#     df_pre = data_preprocessing(df)
#     print("#after preprocessing:", df_pre.shape)
#     df_pre.to_excel(model_path + "\\input_projects.xlsx")
#
#     # split data into train and test
#     X_train, X_test, y_train, y_test = train_test_split(df_pre.abstracts.to_list(), df_pre.disciplines.to_list(),
#                                                         test_size=0.20, random_state=42)
#
#     # select abstracts and discipline to train LLDA model
#     labeled_documents = list(zip(X_train, y_train))
#     print("train abstracts size", len(labeled_documents))
#
#     print("3: train llda model")
#     llda_model = train_llda_model(labeled_documents, alpha, eta, theta_star, common_topic, model_path)
#     print(llda_model.__repr__())
#
#     # print("3.1: evaluate performance of the model")
#     # evaluate_model_performance(llda_model)
#
#     print("4: calculate discipline distance matrix based on topic-term matrix")
#     distance_matrix = create_distance_matrix(llda_model.beta)
#     np.savetxt(model_path + "\\distance_matrix.txt", distance_matrix, fmt="%s")
#
#     print("5: calculate discipline co-occurrence matrix based on doc-topic matrix")
#     occ_matrix = create_occ_matrix(llda_model.theta, 0.1)
#     np.savetxt(model_path + "\\occ_matrix.txt", occ_matrix, fmt="%s")
#
#     print("5.1: create full-network-file for VOSviewer software")
#     create_VOSViewer_network_full(model_path + "\\occ_matrix.txt")
#
#     print("6: calculate IDR (topic diversity) of all projects")
#     # print("Doc-Topic Matrix: \n", llda.theta)
#     # save_doc_topic_matrix(llda_model, model_dir)
#     DT = calculate_topic_diversity(llda_model.theta, distance_matrix)
#
#     tuple_data = list(zip(X_train, y_train, DT))
#     df_result = pd.DataFrame(tuple_data, columns=["Abstracts", "Discipline", "Diversity"])
#     df_IDR_classify = classify_IDR_projects(df_result)
#     df_IDR_classify.to_excel(model_path+"\\result.xlsx")
#
#     # print("7: predict disciplines of test data")
#     # predict_project_disciplines(model_path, X_test, y_test)

# model = load_llda_model('test_data_mapping\\Dimensions_llda_model')
# term_size = len(model.beta[0])
# print(term_size)

