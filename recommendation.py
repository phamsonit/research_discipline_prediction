import numpy as np

# create a dictionary with key as int and value discipline names
from ultil import load_distance_matrix, load_discipline_list


def create_dis_code_table(discipline_list):
    dis_code_table = dict(zip(range(len(discipline_list)), discipline_list))
    return dis_code_table


# map y_test and yhat_trun back to disciplines
def convert_to_discipline(y_test, y_predicted, dis_code_table):
    y_test_disciplines = []
    y_pred_disciplines = []
    if len(y_test) == 0:
        # find disciplines of y_predicted
        pred_tem = []
        for i in range(len(y_predicted)):
            pred_tem = [dis_code_table[x] for x in y_predicted[i]]
        y_pred_disciplines.append(pred_tem)
    else:
        for i in range(len(y_test)):
            # if the y_test have only 1 label
            if type(y_test[i]) == float:
                test_tem = dis_code_table[y_test[i]]
                pred_tem = dis_code_table[y_predicted[i]]
            else:
                # y_test has list of labels
                test_tem = [dis_code_table[x] for x in y_test[i]]
                pred_tem = [dis_code_table[x] for x in y_predicted[i]]
            y_test_disciplines.append(test_tem)
            y_pred_disciplines.append(pred_tem)
    return y_test_disciplines, y_pred_disciplines


# find the distance between two discipline codes based on distance_matrix
# important note: the order of discipline in discipline_list must be the same as they are in the distance matrix
def calculate_distances(y_test, y_predicted, distance_matrix, discipline_list):
    # get number of samples
    sample_size = len(y_test)
    # get number of labels
    # if multilable then nb disciplines is number of label otherwise nb discipline is 1
    if type(y_test[0]) == list:
        label_size = len(y_test[0])
    else:
        label_size = 1
    # create a 2d array to store distances
    discipline_distances = np.ones((sample_size, label_size))
    for i in range(sample_size):
        for j in range(label_size):
            # if the discipline in test data is 'no_discipline' we set distance to other disciplines by 0
            if y_test[i][j] == 'no_discipline' or y_predicted[i][j] == 'no_discipline':
                discipline_distances[i][j] = 0
            else:
                # find the index of disciplines in the list of disciplines
                if label_size == 1:
                    index1 = discipline_list.index(y_test[i])
                    index2 = discipline_list.index(y_predicted[i])
                else:
                    index1 = discipline_list.index(y_test[i][j])
                    index2 = discipline_list.index(y_predicted[i][j])
                # find the distance value from the distance_matrix
                distance = distance_matrix[index1][index2]
                # assign the distance to the output
                discipline_distances[i][j] = distance
    return discipline_distances


# count true positive based on distances of disciplines in y_test and y_predicted
def calculate_tp_distances(discipline_distances, threshold):
    # get number of samples
    sample_size = len(discipline_distances)
    # get number of labels
    label_size = len(discipline_distances[0])
    # count for true positive and err
    sum_tp = 0
    for i in range(sample_size):
        for j in range(label_size):
            # how to choose the threshold?
            if discipline_distances[i][j] <= threshold:
                sum_tp += 1
    ave_tp = sum_tp / (sample_size * label_size)
    #     print('average true positive:', round(ave_tp*100,2))
    return round(ave_tp * 100, 2)


def count_tp_distances(y_test, y_pred, threshold):
    distance_matrix_path = ''
    discipline_list_path = ''
    # load distance matrix
    distance_matrix = load_distance_matrix(distance_matrix_path)
    # load Dimensions discipline list
    discipline_list = load_discipline_list(discipline_list_path)
    # discipline_list
    nb_disciplines = len(discipline_list)
    # create a map between int and discipline-names
    code_table = create_dis_code_table(discipline_list)
    # dis_code_table
    # convert values in y_test and y_predicted back to discipline names
    y_test_disciplines, y_predicted_disciplines = convert_to_discipline(y_test, y_pred, code_table)
    # calculate distances between disciplines in y_test_disciplines and y_predicted_disciplines
    discipline_distances = calculate_distances(y_test_disciplines, y_predicted_disciplines, distance_matrix,
                                               discipline_list)
    # count tp and erro in the discipline distance
    average_tp = calculate_tp_distances(discipline_distances, threshold)
    return average_tp


# count errors by compare one by one between y_test and y_predicted
def count_tp_binary(y_test, y_predicted):
    sum_tp_count = 0
    nb_samples = len(y_test)
    nb_binary_labels = len(y_test[0])
    nb_encoded_labels = sum(y_test[0])
    for i in range(nb_samples):
        for j in range(nb_binary_labels):
            # if both y_test and y_pred are tp increases 1
            if y_test[i][j] == 1 and y_predicted[i][j] == 1:
                sum_tp_count += 1

    M1 = sum_tp_count / (nb_samples * nb_encoded_labels)  #
    #     print('true positive --> %.3f' % round(M2*100,2) )
    return round(M1 * 100, 2)


# count errors by compare number of disciplines matched
# not consider the order of disciplines
def count_tp_int(y_test, y_predicted):
    sum_tp_count = 0
    sample_size = len(y_test)
    label_size = len(y_test[0])
    for i in range(sample_size):
        # find common discipline among two lists
        common_labels = list(set(y_test[i]) & set(y_predicted[i]))
        # number of tp is the number of common labels
        sum_tp_count += len(common_labels)
    M1 = sum_tp_count / (sample_size * label_size)
    # print('average true positive --> %.3f' % round(M1*100,2) )
    return round(M1 * 100, 2)


# count err for each class
def count_tp_int_class(y_test, y_predicted, nb_disciplines):
    nb_samples = len(y_test)
    count_disciplines = dict(zip(range(nb_disciplines), [0] * nb_disciplines))
    for i in range(nb_samples):
        count_disciplines[y_test[i]] = count_disciplines[y_test[i]] + 1
    #     print('number of disciplines')
    #     print(count_disciplines)

    count_tp = dict(zip(range(nb_disciplines), [0] * nb_disciplines))
    for i in range(nb_samples):
        if y_test[i] == y_predicted[i]:
            count_tp[y_test[i]] = count_tp[y_test[i]] + 1
    #     print('true positive')
    #     print(count_tp)

    count_tn = dict(zip(range(nb_disciplines), [0] * nb_disciplines))
    for i in range(nb_samples):
        if y_test[i] != y_predicted[i]:
            count_tn[y_test[i]] = count_tn[y_test[i]] + 1
    #     print('true negative')
    #     print(count_tn)

    count_acc = dict(zip(range(nb_disciplines), [0] * nb_disciplines))
    for i in range(nb_disciplines):
        if count_disciplines[i] > 0:  # discipline not exits in test data
            count_acc[i] = round(count_tp[i] / count_disciplines[i], 2)
    #     print('accuracy')
    #     print(count_acc)

    count_err = dict(zip(range(nb_disciplines), [0] * nb_disciplines))
    for i in range(nb_disciplines):
        if count_disciplines[i] > 0:
            count_err[i] = round(count_tn[i] / count_disciplines[i], 2)
    #     print('err')
    #     print(count_err)

    print('id \t #dis \t tp \t tn \t acc \t err')
    sum_tp = 0
    for i in range(nb_disciplines):
        print(i, '\t', count_disciplines[i], '\t', count_tp[i], '\t', count_tn[i], '\t', count_acc[i], '\t',
              count_err[i])
        if count_acc[i] >= 0.5:
            sum_tp += 1

    print('sum tp', sum_tp)

    return count_acc, count_err


def count_tp_with_distance_matrix(y_test, y_pred, distance_threshold, distance_matrix, discipline_list):
    # y_test = [[2, 5],[...]]
    # y_pred = [[2, 4, 6], [...]]
    sample_size = len(y_test)
    sum_tp = 0
    for s in range(sample_size):
        # for each sample count average tp
        y_test_s = list(y_test[s])
        y_pred_s = list(y_pred[s])
        if len(y_pred_s) > 0:
            tp = 0
            for p in y_pred_s:
                # if p in y_test = tp count
                if p in y_test_s:
                    tp += 1
                    # remove p from y_test list
                    y_test_s.remove(p)
                else:
                    # print(p)
                    # find distances between p with other disciplines in distance matrix
                    distances = list(distance_matrix[int(p)])
                    # remove distance p itself
                    distances.remove(0)
                    # find minimun distance between p and others
                    min_distance = min(distances)
                    if min_distance < distance_threshold:
                        # find disicpline ids
                        closest_disciplines = [i for i, j in enumerate(distances) if j == min_distance]
                        # print(closest_disciplines)
                        # if one of closest disciplines is in y_test the count tp
                        for cd in closest_disciplines:
                            if cd in y_test_s:
                                tp += 1
                                y_test_s.remove(cd)
                                break;
            # calculate average tp for this sample
            # print(tp)
            sum_tp += tp / len(y_pred_s)
    # calculate average tp for all samples
    avg_tp = sum_tp / sample_size

    return avg_tp * 100

##### RECOMMENDATION PROCEDURES ######

def find_close_disciplines(pred_disciplines, distance_matrix, discipline_list, display):
    r_dis = list()
    if display:
        print('# Close disciplines:')
    for d in pred_disciplines:
        # find index of d in discipline list
        row_index = discipline_list.index(d)
        # find distances of row_index in distance matrix
        distances = distance_matrix[row_index]
        # creat list of tuple(discipline name, distances)
        distances_tuple = list(zip(discipline_list, distances))
        # sort the list by simiarility of the discipline
        distances_tuple.sort(key=lambda t: t[1])
        # add d
        r_dis.append(distances_tuple[0])
        # add 2 other disciplines closest to d
        temp = distances_tuple[1:3]
        r_dis.extend(temp)
        if display:
            display_disciplines(temp, '- close to %s:' % d)
    return r_dis


# find closest disciplines with disciplines from y_pred
def recommend_disciplines(y_pred, distance_matrix, discipline_list, display):
    # create a map between int and discipline-names
    code_table = create_dis_code_table(discipline_list)
    # get all predicted disciplines
    pred_disciplines = [code_table[x] for x in y_pred]
    if display:
        display_disciplines(pred_disciplines, '# Predicted disciplines:')
    # print('# predicted disciplines:', pred_disciplines)
    # find close disciplines with disciplines in y_predicted_disciplines
    close_disciplines = find_close_disciplines(pred_disciplines, distance_matrix, discipline_list, display)
    # filter out duplicates
    rd_filter = filter_recommended_disciplines(close_disciplines)
    # print recommended disciplines
    if display:
        display_disciplines(rd_filter, '# Recommended disciplines:')
    # calcualte discipline diversity
    return rd_filter


def display_disciplines(d, msg):
    print(msg)
    for i in d:
        print('--', i)


# filter out dup disciplines in the list of recommended disciplines
def filter_recommended_disciplines(rd):
    # select the lowest distance if there are dup
    rd_filter = dict()
    for r in rd:
        if r[0] not in rd_filter.keys():
            rd_filter[r[0]] = 1 - r[1]
        else:
            # if discipline exists in the dict then select the smaller distance
            if (1 - r[1]) > rd_filter[r[0]]:
                rd_filter[r[0]] = 1 - r[1]
    return rd_filter


# create discipline vector for recommended disciplines
def create_predicted_discipline_vector(rd_filter, discipline_list):
    discipline_size = len(discipline_list)
    distance_sum = sum(rd_filter.values())
    # print(sum_distance)
    # create distances vector such that sum = 1
    discipline_vector = np.zeros(discipline_size)
    for key, value in rd_filter.items():
        index = discipline_list.index(key)
        # normalise distance
        discipline_vector[index] = value / distance_sum
    return discipline_vector

def calculate_diversity_vector(discipline_vector, distance_matrix):
    """
    # calculate diversity of attached disciplines
    :param discipline_vector: attached discipline vector
    :param distance_matrix: distance matrix
    :return: rao-stirling index score
    """
    rs = 0
    size = len(discipline_vector)
    for i in range(size):
        for j in range(size):
            rs += discipline_vector[i] * discipline_vector[j] * distance_matrix[i][j]
    return rs

# calculate topic diversity based on Rao-Stirling diversity
def calculate_predicted_discipline_diversity(rd, distance_matrix, discipline_list):
    # rs = sum fi*fj*d(i,j) where fi, fj: probability; d(i,j): distance between i and j
    # create discipline vector
    discipline_vector = create_predicted_discipline_vector(rd, discipline_list)
    # calculate diversity
    rs = calculate_diversity_vector(discipline_vector, distance_matrix)
    return round(rs, 2)


def predict_disciplines(sample, t2v_model, ml_model, llda_model, distance_matrix, discipline_list, display):
    # search topic probability distribution over the sample
    topics_words, word_scores, topic_scores, topic_nums = t2v_model.query_topics(sample, t2v_model.get_num_topics())

    # sort the order of topics and scores
    l_tuples = list(zip(topic_scores, topic_nums))
    l_tuples.sort(key=lambda y: y[1])
    # get topic probability distribution
    sample_topic_proba = [x[0] for x in l_tuples]

    # predict label of the sample data
    y_pred = ml_model.predict([sample_topic_proba])

    # find discipline probability by L-LDA
    # dis_proba = predict_discipline_proba(sample, llda_model, discipline_list)

    # recommend close disciplines
    rec_dis = recommend_disciplines(y_pred[0], distance_matrix, discipline_list, display)

    return rec_dis
