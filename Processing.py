from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# data_processing
def data_processing_credit(data, remove_rate):
    # remove outlier
    data = data.dropna()
    data = data.drop_duplicates()
    data.reset_index(inplace=True)
    # encode and change data type
    key_encoder = LabelEncoder()
    data["person_home_ownership"] = key_encoder.fit_transform(data["person_home_ownership"])
    data["loan_intent"] = key_encoder.fit_transform(data["loan_intent"])
    data["loan_grade"] = key_encoder.fit_transform(data["loan_grade"])
    data["cb_person_default_on_file"] = key_encoder.fit_transform(data["cb_person_default_on_file"])
    data["person_age"] = pd.to_numeric(data["person_age"])
    data["person_income"] = pd.to_numeric(data["person_income"])
    data["person_emp_length"] = pd.to_numeric(data["person_emp_length"])
    data = data.drop("index", axis=1)

    # remove 90% of default cases as default cases are very rare in real world
    print(data['loan_status'].value_counts())
    data1 = data.drop(data.query('loan_status == 1').sample(frac=.6, random_state=1).index)
    default_case = data[~data.index.isin(data1.index)]
    data1 = data1.drop(data1.sample(frac=remove_rate, random_state=1).index)
    print(data1['loan_status'].value_counts())
    data2 = data1.drop(data1.sample(frac=0.3, random_state=1).index)
    data2 = data2.drop(data2.query('loan_status == 1').sample(frac=.9, random_state=1).index)
    print(data2['loan_status'].value_counts())
    return data1, default_case, data2

def data_processing_music(data):
    # remove outlier
    data = data.dropna()
    data = data.drop_duplicates()
    genre=sorted(data['music_genre'].unique())[11:]
    data=data[data['music_genre'].isin(genre)]
    data=data[data["tempo"] != "?"]
    data.reset_index(inplace=True)

    data = data.drop(data.sample(frac=.3).index)

    print(data['music_genre'].value_counts())
    # print(data['key'].value_counts())
    # print(data['mode'].value_counts())
    # encode and change data type
    key_encoder = LabelEncoder()
    data["key"] = key_encoder.fit_transform(data["key"])
    data["mode"] = key_encoder.fit_transform(data["mode"])
    data = data.drop("instance_id", axis=1)
    data = data.drop("obtained_date", axis=1)
    data = data.drop("index", axis=1)

    data["popularity"] = pd.to_numeric(data["popularity"])
    data["acousticness"] = pd.to_numeric(data["acousticness"])
    data["danceability"] = pd.to_numeric(data["danceability"])
    data["duration_ms"] = pd.to_numeric(data["duration_ms"])
    data["energy"] = pd.to_numeric(data["energy"])
    data["instrumentalness"] = pd.to_numeric(data["instrumentalness"])
    data["liveness"] = pd.to_numeric(data["liveness"])
    data["loudness"] = pd.to_numeric(data["loudness"])
    data["speechiness"] = pd.to_numeric(data["speechiness"])
    data["tempo"] = pd.to_numeric(data["tempo"])
    data["valence"] = pd.to_numeric(data["valence"])

    # create imbalance dataset
    data1 = data.drop(data.query('music_genre == "Electronic"').sample(frac=.6, random_state=1).index)
    data1 = data1.drop(data1.query('music_genre == "Rock"').sample(frac=.3, random_state=1).index)
    data1 = data1.drop(data1.query('music_genre == "Hip-Hop"').sample(frac=.4, random_state=1).index)
    data1 = data1.drop(data1.query('music_genre == "Anime"').sample(frac=.5, random_state=1).index)
    data1 = data1.drop(data1.query('music_genre == "Country"').sample(frac=.7, random_state=1).index)
    print(data1['music_genre'].value_counts())
    return data, data1

def plot_counts(data, feature, order=None):
    sns.countplot(x=feature, data=data, palette="ocean", order=order)
    plt.title(f"Counts in each {feature}")
    plt.xticks(rotation=30)
    plt.savefig(feature+'.png')