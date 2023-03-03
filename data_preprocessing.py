import numpy as np
import pandas as pd
from numpy import arange
from sklearn.preprocessing import MultiLabelBinarizer


# TODO: define procedures for filtering project data

def create_binary_label(df):
    mlb = MultiLabelBinarizer()
    binary_label = mlb.fit_transform(df.disciplines)
    dis_codes = mlb.classes_
    # return encoded binary matrix and names of disciplines
    return binary_label, dis_codes


def create_int_label(df_new_analysis, dis_codes, nb_new_disciplines):
    nb_projects = df_new_analysis.shape[0]
    nb_org_disciplines = len(dis_codes)

    # create coding table: a dictionary where key = discipline, value = int
    dis_code_table = dict(zip(dis_codes, arange(0, nb_org_disciplines)))

    # create 2d array and assign all value to nb_org_disciplines, e.g., 42
    int_label = np.ones((nb_projects, nb_new_disciplines)) * nb_org_disciplines
    # iterate through each row and select
    # 'disciplines' to encode.
    for i in range(nb_projects):
        org_dis = df_new_analysis.loc[i, 'disciplines']
        # for each discipline
        for j in range(len(org_dis)):
            # get discipline name
            org_dis_name = org_dis[j]
            # find id of discipline in dis_code_table and assign to new_dis_code
            int_label[i][j] = dis_code_table[org_dis_name]
    return int_label


def load_VODS(file_name):
    df = pd.read_excel(file_name, converters={'Code_Level_2': str})
    codes = df['Code_Level_2'].to_list()
    names = df['Label_Level_2'].to_list()
    dic = dict(zip(codes, names))
    return dic


def create_code_dis_table(dis_codes):
    # save discipline code table
    nb_dis = len(dis_codes)
    # if number of encode disciplines is larger than number of discipline in project
    # assign a `dummy-discipline' to the missing label
    dis_codes_tem = list(dis_codes)
    dis_codes_tem.append("dummy-discipline")

    range_values_temp = [x for x in range(nb_dis)]
    range_values_temp = [round((x / nb_dis), 2) for x in range_values_temp]

    dis_code_table_temp = dict(zip(range_values_temp, dis_codes_tem))
    return dis_code_table_temp


def label_encoding(df, database, output_path):
    # create binary matrix
    binary_label, label_names = create_binary_label(df)
    # save binary matrix to disk
    np.savetxt(output_path+'%s_binary_label.txt' % database, binary_label, fmt='%s')
    # save label name to disk
    if database == 'FRIS':
        try:  # if this file is not available we will use discipline codes as they are in the FRIS data
            vods = load_VODS(
                'C:\\Users\\lucp11046\\PycharmProjects\\IDRProject\\codes_translation\\data\\VODS_level_2_categories.xlsx')
            if vods is not None:
                fris_dis_codes = [x + ' ' + vods[x] for x in label_names]
                fris_dis_codes.append('no_discipline')
                np.savetxt(output_path+'%s_discipline_list.txt' % database, fris_dis_codes, fmt='%s')
        except:
            np.savetxt(output_path + '%s_discipline_list.txt' % database, label_names, fmt='%s')
    else:
        np.savetxt(output_path+'%s_discipline_list.txt' % database, label_names, fmt='%s')

    # create int label
    int_label = create_int_label(df, label_names, 2)
    # save to disk
    np.savetxt(output_path+'%s_int_label.txt' % database, int_label, fmt='%s')

    return int_label


