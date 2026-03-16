import pandas
import pandas as pd
import  matplotlib.pyplot as plt
import  matplotlib
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
import os

matplotlib.use('TkAgg')




def IQR_cleaning(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df[column] <= Q3 + (IQR * 1.5)]
    df = df[df[column] >= Q1 - (IQR * 1.5)]

    return df

def boxplot(df, column):
    fig, axs = plt.subplots(figsize=(12, 6))
    sns.boxplot(df[column])
    plt.show()



def df_info(df):
    print(df.head())
    print(df.describe().transpose())
    print("Количество строк: ", len(df))
    print("Количество NAN:\n", df.isna().sum())



def clean_data(path):
    df = pd.read_csv(path)
    df_info(df)

    df = df.dropna()
    df = IQR_cleaning(df, 'avg_systolic')
    df = IQR_cleaning(df, 'avg_diastolic')

    print("После очистки")
    df_info(df)
    new_name = path[:-4] + '_clean.csv'
    df.to_csv(new_name, index=False)
    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled



def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def save_df(df, path):
    df = pandas.DataFrame(df)
    df.to_csv(path, index=False)


def save_scaled_df(X_train,y_train, X_test, y_test):
    working_dir = os.getcwd()
    path_test = os.path.join(working_dir, 'test')
    path_train = os.path.join(working_dir, 'train')

    if not os.path.isdir(path_test):
        os.makedirs('test', exist_ok=True)

    if not os.path.isdir(path_train):
        os.makedirs('train', exist_ok=True)


    save_df(X_train, os.path.join(path_train, 'X.csv'))
    save_df(y_train, os.path.join(path_train, 'y.csv'))

    save_df(X_test, os.path.join(path_test, 'X.csv'))
    save_df(y_test, os.path.join(path_test, 'y.csv'))


if __name__ == "__main__":
    print("Очистка данных")
    # Возможные датасеты train
    # train/train0.csv
    # train/train1.csv
    # train/train2.csv
    # Возможные датасеты test
    # test/test0.csv
    # test/test1.csv
    # test/test2.csv
    path ='train/train0.csv'
    path_test ='test/test0.csv'

    df = clean_data(path)
    df_test = clean_data(path_test)

    print("")

    X_train, y_train = split_data(df, target='hypertension')
    X_test, y_test = split_data(df, target='hypertension')

    X_train, X_test =  scale_data(X_train, X_test)

    save_scaled_df(X_train,y_train, X_test, y_test)







