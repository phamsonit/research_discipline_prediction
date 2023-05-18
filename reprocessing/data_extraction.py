from select_projects import select_projects
from metadata_based_framework.text_cleaning import clean_all

################ extract grants data from Dimensions #################
import dimcli
import sys
import pandas as pd

if not 'google.colab' in sys.modules:
  # make js dependecies local / needed by html exports
  from plotly.offline import init_notebook_mode
  init_notebook_mode(connected=True)


def connect_dimensions():
    print("==\nLogging in..")
    # https://digital-science.github.io/dimcli/getting-started.html#authentication
    ENDPOINT = "https://app.dimensions.ai"
    if 'google.colab' in sys.modules:
        import getpass
        KEY = getpass.getpass(prompt='API Key: ')
        dimcli.login(key=KEY, endpoint=ENDPOINT)
    else:
        KEY = ""
        dimcli.login(key=KEY, endpoint=ENDPOINT)
    dsl = dimcli.Dsl()
    return dsl


# extract research id
def get_researcher_id(llist):
    if type(llist) == list:
        re_ids = []
        for i in llist:
            re_ids.append(i['id'])
        return re_ids


# method to keep only the 4 digits FOR category
def get_4_digits_cat(llist):
    if type(llist) == list:
        for d in llist:
            c = d['name'].split()
            if len(c[0]) == 4:
                return d['name']
    return ""


# method to get all unique 4 digit FOR from result
def get_unique_FOR_cat_list(category_for):
    unique_for = category_for.apply(lambda x: get_4_digits_cat(x))
    unique_for_drop_dup = unique_for.drop_duplicates()
    unique_for_sort = unique_for_drop_dup.sort_values().to_list()
    return unique_for_sort


# method to keep only the 4 digits FOR category
def extract_4_digits_FOR_cat(for_cats):
    FOR_list = []
    if type(for_cats) == list:
        for d in for_cats:
            c = d['name'].split()
            if len(c[0]) == 4:
                FOR_list.append(d['name'])
    return FOR_list


def extract_relevant_concepts(concepts):
    concepts_list = []
    if type(concepts) == list:
        for c in concepts[:10]:
            concepts_list.append(c["concept"])
    return concepts_list


def extract_dimensions_grants_data():
    dsl = connect_dimensions()

    funders = """["European Commission", "Belgian Federal Science Policy Office", "Research Foundation - Flanders", "Fund for Scientific Research"]"""
    country = "\"Belgium\""
    start_year = "2010"
    query = f"""
        search grants
        where researchers is not empty and start_year >= {start_year}
        and funder_org_name in {funders}
        and research_org_countries.name={country}
        return grants[id+title+abstract+category_for_2020+concepts_scores]
        """

    res = dsl.query_iterative(query)
    # res = dsl.query(query)

    # concat result to dataframe
    df = res.as_dataframe()

    df = df.dropna(subset=["abstract"])

    df = df[df.abstract.apply(lambda x: len(x.split(' ')) > 100)].copy()

    # extract disciplines = [4 digits of FOR]
    df['disciplines'] = df.category_for_2020.apply(lambda x: extract_4_digits_FOR_cat(x))

    # filter out projects have less than 2 disciplines
    df = df[df.disciplines.apply(lambda x: len(x) > 1)].copy()

    # create keywords form relevent concepts
    df['keywords'] = df.concepts_scores.apply(lambda x: extract_relevant_concepts(x))

    df.reset_index(drop=True)

    df.to_csv("project_data\\Dimensions_projects_data.csv")

    return df


def clean_project_data(database_name):
    if database_name == "FRIS":
        df = pd.read_csv("../project_data/FRIS_projects_data.csv")
    else:
        df = pd.read_csv("../project_data/Dimensions_projects_data.csv")

    print("before clean data", df.shape)
    # concat title and abstract as input of the model
    df['abstracts'] = df.title.str.cat(df.abstract, sep=' ')
    df_clean = clean_all(df, "abstracts")
    df_clean.reset_index()
    print("after clean data", df_clean.shape)
    df_clean.to_csv("project_data\\"+database_name+"_projects_data_clean.csv")
    return df_clean


# FRIS data are already fetched and stored in files
def extract_fris_data(start_year, end_year, code_level):
    df = select_projects.select_project_details(start_year, end_year, code_level)
    # select project which have more than 200 words
    df_filter = df[df.abstract.apply(lambda x: len(x.split(' ')) > 100)].copy()
    # select project which have more than 1 assigned disciplines
    df_filter1 = df_filter[df_filter.disciplines.apply(lambda x: len(x) > 1)].copy()
    df_filter1.to_csv("project_data\\FRIS_projects_data.csv")
    return df_filter1.reset_index()

# extract_dimensions_grants_data()
# clean_project_data("Dimensions")
