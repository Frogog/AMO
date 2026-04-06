import numpy as np
import pandas as pd
import os


def create_measurments_folder():
    os.makedirs('test', exist_ok=True)
    os.makedirs('train', exist_ok=True)


def generate_noize(df):
    for column in df.columns:
        mask = np.random.random(len(df)) < 0.01
        df[column] = df[column].mask(mask)

    return df

def generate_anomaly(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        outlier_mask = np.random.random(len(df)) < 0.02

        if column == 'age':
            df.loc[outlier_mask, column] = np.random.choice([5, 100, 110], outlier_mask.sum())
        elif column in ['avg_systolic', 'avg_diastolic']:
            df.loc[outlier_mask, column] = np.random.choice([50, 220, 250], outlier_mask.sum())

    return df

def generate_measurements_data(n_samples=1000, random_seed = None):
    # normal_people = int(n_samples*0.95)
    # anomaly_people = int(n_samples - normal_people)

    if random_seed is not None:
        np.random.seed(random_seed)

    age = np.random.randint(18, 81, n_samples)
    sex = np.random.randint(0, 2, n_samples)

    base_systolic = 100 + age * 0.5 + np.random.normal(0, 8, n_samples)
    base_diastolic = 65 + age * 0.2 + np.random.normal(0, 5, n_samples)

    systolic = base_systolic + np.random.normal(0, 5, n_samples)
    diastolic = base_diastolic + np.random.normal(0, 3, n_samples)

    hypertension = np.where((systolic >= 140) | (diastolic >= 90), 1, 0)

    borderline = (systolic >= 135) & (systolic < 140) & (diastolic < 90)
    hypertension[borderline] = np.random.choice([0, 1], borderline.sum(), p=[0.5, 0.5])

    systolic = np.round(systolic).astype(int)
    diastolic = np.round(diastolic).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'avg_systolic': systolic,
        'avg_diastolic': diastolic,
        'hypertension': hypertension
    })

    df = generate_noize(df)
    df = generate_anomaly(df)

    print(df.head())
    print(df['sex'].value_counts())
    print(df['hypertension'].value_counts())
    print(df.count())

    return df

def generate_all_data(n_parts, n_samples, test_split):
    create_measurments_folder()
    for i in range(n_parts):
        n_train = int(n_samples * (1 - test_split))
        n_test = int(n_samples * test_split)
        generate_measurements_data(n_train).to_csv(f"train/train{i}.csv", index=False)
        generate_measurements_data(n_test).to_csv(f"test/test{i}.csv", index=False)


def check_folder_data(folder_path, file_name):
    files = []
    for i in range(1000):
        filename = f"{file_name}{i}.csv"
        file_path = os.path.join(folder_path, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            files.append((filename, len(df)))

        if i > 10 and not os.path.exists(file_path):
            break

    print(f"\n{file_name} файлы:")
    if files:
        for filename, rows in files:
            print(f"   {filename}: {rows} строк")

    return files


def check_all_data():
    print(f"\nПроверка папки с данными для обучения")

    train_files = check_folder_data("train","train")
    test_files = check_folder_data("test","test")

    total_files = len(train_files) + len(test_files)
    total_rows = sum(r for _, r in train_files) + sum(r for _, r in test_files)
    print(f"Статистичка по данным для обучения: {total_files} файлов, {total_rows} строк")

if __name__ == "__main__":
    print("Генерация данных")
    generate_all_data(3, 1000, 0.1)
    check_all_data()

