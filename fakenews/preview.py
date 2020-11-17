''' A simple data pre-visualisation with pandas '''
import pandas as pd
import matplotlib.pyplot as plt


DATAPATH_FAKE = '~/MEGA/researches/databases/fake-news/Fake.csv'
DATAPATH_REAL = '~/MEGA/researches/databases/fake-news/True.csv'


def is_missing(row):
    ismissing = is_empty(row['title'])
    ismissing = ismissing or is_empty(row['text'])
    ismissing = ismissing or is_empty(row['subject'])
    ismissing = ismissing or is_empty(row['date'])
    return ismissing


def is_empty(value):
    return value == '' or value.strip() == ''


if __name__ == '__main__':
    data_fake = pd.read_csv(DATAPATH_FAKE)
    data_real = pd.read_csv(DATAPATH_REAL)

    # print(data_fake.head())
    # print(data_fake.axes)
    # print(data_real)

    # Preview the values and counts of subjects for FAKE data
    # print(data_fake.groupby(['subject']).count()['title'])
    # Preview the values and counts of subjects for REAL data
    # print(data_real.groupby(['subject']).count()['title'])

    # Finding missing data
    # print(data_real.head())
    missing_rows = []
    for index, row in data_fake.iterrows():
        if is_missing(row):
            missing_rows.append(index)

    # print(np.array(missing_rows))

    print(data_fake.iloc[missing_rows]['text'])

    dist_real = data_real.groupby(['subject']).count()['title']
    # Preview of subject distribution
    # plt.bar(dist_real.axes[0], dist_real.values, color=['red', 'blue'])
    # plt.show()

    dist_fake = data_fake.groupby(['subject']).count()['title']
    plt.bar(dist_fake.axes[0], dist_fake.values, color=['red', 'blue'])
    plt.xlabel('Values', fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.xticks(fontsize=17, rotation=90)
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.show()
