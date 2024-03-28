import numpy as np
import pandas as pd


def generate_dataframe() -> pd.DataFrame:
    keys = ['row.names', 'pclass', 'survived', 'name', 'age', 'embarked', 'sex']

    places = set()

    initial_dict_list = []

    file = open('data/dataset.txt', 'r', encoding='utf-8')
    i = 0
    value_list = []
    for line in file:
        if i > 0:
            # remove suffix '\n'
            row = line.removesuffix('\n')
            # add row.names
            row_names = row[0:line.find(",")].strip('"')
            row = row.replace('"' + row_names + '",', '')
            value_list.append(row_names)

            # add pclass
            pclass = row[0:6].removesuffix(',').strip('"').strip(' ')
            row = row[6:]
            value_list.append(pclass)

            # add survived
            survived = row[0:2].removesuffix(',').strip(' ')
            row = row[2:]
            value_list.append(survived)

            # add name
            name = row[0:row.find('",')+2]
            row = row.replace(name, '')
            value_list.append(name.strip(',').strip('"').strip(' '))

            # add age
            age = row[0:row.find(',')+1]
            row = row.replace(age, '')
            value_list.append(age.strip(',').strip('"').strip(' '))

            # add sex and embarked
            sex = row[row.rfind(',')+1:]
            row = row.replace(sex, '').strip(',').strip('"').replace('"', '').replace(",,,", ",")

            # add embarked in places set
            # if place is empty, reset value_list and skip
            if len(row) == 0:
                value_list = []
                continue
            places.add(row)
            value_list.append(row)
            value_list.append(sex.strip('"'))
            # check first 104 list value
            # if i <= 104:
            #     print(value_list)
            res = dict(zip(keys, value_list))
            # check first 104 dict value
            # if i <= 104:
            #     print(res)
            initial_dict_list.append(res)
            value_list = []
        i += 1

    df = pd.DataFrame.from_dict(initial_dict_list)
    df['pclass'] = df['pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['survived'] = df['survived'].map({'0': 0, '1': 1})
    age_sum = np.double(0)
    na_count = 0
    for age in df['age']:
        if age != 'NA':
            age = np.double(float(age))
            age_sum += age
        else:
            na_count += 1
    age_mean = np.double(age_sum / (550 - na_count))
    # print(age_mean)

    for i in df.index:
        if df.at[i, 'age'] != 'NA':
            df.at[i, 'age'] = np.double(float(df.at[i, 'age']))
        else:
            df.at[i, 'age'] = age_mean

    df['age'] = df['age'].astype(np.double)

    place_map = {}
    i = 0
    j = -1
    for place in places:
        if place.count("Southampton") >= 1:
            if j < 1:
                i += 1
                j = i
            place_map.update({place: j})
        else:
            i += 1
            place_map.update({place: i})
    # print(place_map)
    df['embarked'] = df['embarked'].map(place_map)
    df.drop(['name', 'row.names'], axis=1, inplace=True)
    return df
