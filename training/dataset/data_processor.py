import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from dataset.data_utils import SummaryWritter



# Returns proxy column names given protected columns and noise param.
def get_proxy_column_names(protected_columns, noise_param):
    return ['PROXY_' + '%0.2f_' % noise_param + column_name for column_name in protected_columns]

def generate_proxy_columns(df, protected_columns, noise_param=1):
    """Generates proxy columns from binarized protected columns.

    Args: 
      df: pandas dataframe containing protected columns, where each protected 
        column contains values 0 or 1 indicating membership in a protected group.
      protected_columns: list of strings, column names of the protected columns.
      noise_param: float between 0 and 1. Fraction of examples for which the proxy 
        columns will differ from the protected columns.

    Returns:
      df_proxy: pandas dataframe containing the proxy columns.
      proxy_columns: names of the proxy columns.
    """
    proxy_columns = get_proxy_column_names(protected_columns, noise_param)
    num_datapoints = len(df)
    num_groups = len(protected_columns)
    noise_idx = random.sample(range(num_datapoints), int(noise_param * num_datapoints))
    proxy_groups = np.zeros((num_groups, num_datapoints))
    df_proxy = df.copy()
    for i in range(num_groups):
        df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
    for j in noise_idx:
        #for single group
        #'female' is 0, 'male' is 1
        if df_proxy[proxy_columns[0]][j] == 'Male':
            df_proxy.at[j, proxy_columns[0]] = 'Female'
            break
    return df_proxy

def generate_proxy_columns_bank(df, protected_columns, noise_param=1):
    """Generates proxy columns from binarized protected columns.

    Args: 
      df: pandas dataframe containing protected columns, where each protected 
        column contains values 0 or 1 indicating membership in a protected group.
      protected_columns: list of strings, column names of the protected columns.
      noise_param: float between 0 and 1. Fraction of examples for which the proxy 
        columns will differ from the protected columns.

    Returns:
      df_proxy: pandas dataframe containing the proxy columns.
      proxy_columns: names of the proxy columns.
    """
    proxy_columns = get_proxy_column_names(protected_columns, noise_param)
    num_datapoints = len(df)
    num_groups = len(protected_columns)
    noise_idx = random.sample(range(num_datapoints), int(noise_param * num_datapoints))
    # print(noise_idx, '1111111111111111')
    proxy_groups = np.zeros((num_groups, num_datapoints))
    df_proxy = df.copy()
    for i in range(num_groups):
        df_proxy[proxy_columns[i]] = df_proxy[protected_columns[i]]
    for j in noise_idx:
        #for single group
        #<25 or >60 is 0, [25,60] is 1
        if df_proxy[proxy_columns[0]][j] >=25 and df_proxy[proxy_columns[0]][j] <=60:
            df_proxy.at[j, proxy_columns[0]] = 65
            break
    return df_proxy

def load_adult_data(csvfile, group="sex", ratio=0.1, seed=42, onehot=True, noise_param=0.5):
    df = pd.read_csv(csvfile)

    # Compute true group labels
    if group == "sex":
        Z_all = np.array([0 if z == "Female" else 1 for z in df["sex"]], dtype=np.float32)
    elif group == "race":
        Z_all = np.array([0 if z != "White" else 1 for z in df["race"]], dtype=np.float32)
    elif group == "age":
        Z_all = np.array([0 if (z < 25) | (z > 60) else 1 for z in df["age"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y_all = np.array([1 if ">50K" in y else 0 for y in df['income']], dtype=np.float32)
    df_features = df.drop(columns=['income'])

    col_quali = list(df_features.select_dtypes(include='O').columns)
    col_quanti = list(df_features.select_dtypes(include='int').columns)

    X_quali = df_features[col_quali].values
    X_quanti = df_features[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X_all = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    # Split before flipping
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, df_train, df_test = train_test_split(
        X_all, Y_all, Z_all, df, test_size=ratio, random_state=seed, stratify=np.vstack((Y_all, Z_all)).T)

    # Now flip proxy labels in train only
    protected_columns = ['sex']
    df_train_flipped = generate_proxy_columns(df_train, protected_columns, noise_param=noise_param)

    if group == "sex":
        Z_train_proxy = np.array([0 if z == "Female" else 1 for z in df_train_flipped["PROXY_%.2f_sex" % noise_param]], dtype=np.float32)
        Z_test_proxy = np.array([0 if z == "Female" else 1 for z in df_test["sex"]], dtype=np.float32)  # test set not flipped
    elif group == "race":
        Z_train_proxy = np.array([0 if z != "White" else 1 for z in df_train_flipped["race"]], dtype=np.float32)
        Z_test_proxy = np.array([0 if z != "White" else 1 for z in df_test["race"]], dtype=np.float32)
    elif group == "age":
        Z_train_proxy = np.array([0 if (z < 25) | (z > 60) else 1 for z in df_train_flipped["age"]], dtype=np.float32)
        Z_test_proxy = np.array([0 if (z < 25) | (z > 60) else 1 for z in df_test["age"]], dtype=np.float32)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train_proxy)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train_proxy)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)


def load_bank_data(csvfile, group="age", ratio=0.2, seed=42, onehot=True, noise_param=0.1):
    # It is bank marketing data.
    # bank.csv 462K lines 450 Ko
    # bank-full 4M 614K lines 4.4 Mo
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    
    df = pd.read_csv(csvfile)

    # Compute true group labels
    if group == "marital":
        Z_all = np.array([0 if z == "non-married" else 1 for z in df["marital"]], dtype=np.float32)
    elif group == "age":
        Z_all = np.array([0 if (z < 25) | (z > 60) else 1 for z in df["age"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y_all = np.array([1 if y == 'yes' else 0 for y in df['y']], dtype=np.float32)
    df_features = df.drop(columns=['y'])

    col_quali = list(df_features.select_dtypes(include='O').columns)
    col_quanti = list(df_features.select_dtypes(include='int').columns)

    X_quali = df_features[col_quali].values
    X_quanti = df_features[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X_all = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    # Split before flipping
    X_train, X_test, Y_train, Y_test, Z_train, Z_test, df_train, df_test = train_test_split(
        X_all, Y_all, Z_all, df, test_size=ratio, random_state=seed, stratify=np.vstack((Y_all, Z_all)).T)

    # Now flip protected columns in train only
    protected_columns = [group]
    df_train_flipped = generate_proxy_columns_bank(df_train, protected_columns, noise_param=noise_param)

    if group == "marital":
        Z_train_proxy = np.array([0 if z == "non-married" else 1 for z in df_train_flipped["PROXY_%.2f_marital" % noise_param]], dtype=np.float32)
        Z_test_proxy = np.array([0 if z == "non-married" else 1 for z in df_test["marital"]], dtype=np.float32)
    elif group == "age":
        Z_train_proxy = np.array([0 if (z < 25) | (z > 60) else 1 for z in df_train_flipped["PROXY_%.2f_age" % noise_param]], dtype=np.float32)
        Z_test_proxy = np.array([0 if (z < 25) | (z > 60) else 1 for z in df_test["age"]], dtype=np.float32)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train_proxy)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train_proxy)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)


def load_default_data(csvfile, group="sex", ratio=0.2, seed=42, onehot=True, noise_param=0.9):
    df = pd.read_csv(csvfile)

    if group == "sex":
        Z_all = np.array([0 if z == "Female" else 1 for z in df["sex"]], dtype=np.float32)
    else:
        raise ValueError("Invalid group name!")

    Y_all = np.array([1 if y == 1 else 0 for y in df['default payment next month']], dtype=np.float32)

    df_features = df.drop(columns=['default payment next month'])

    col_quanti = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
                  'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                  'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                  'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    col_quali = ['sex', 'EDUCATION', 'MARRIAGE']

    X_quali = df_features[col_quali].values
    X_quanti = df_features[col_quanti].values

    if onehot:
        quali_encoder = OneHotEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali).toarray()
    else:
        quali_encoder = OrdinalEncoder(categories="auto")
        X_quali = quali_encoder.fit_transform(X_quali)

    X_all = np.concatenate([X_quanti, X_quali], axis=1, dtype=np.float32)

    # Train-test split first (based on true group)
    X_train, X_test, Y_train, Y_test, Z_train_true, Z_test = train_test_split(
        X_all, Y_all, Z_all, test_size=ratio, random_state=seed, stratify=np.vstack((Y_all, Z_all)).T)

    # Also split the raw DataFrame for flipping
    df_train, df_test = train_test_split(
        df, test_size=ratio, random_state=seed, stratify=np.vstack((Y_all, Z_all)).T)

    # Flip group labels in training set only
    protected_columns = ['sex']
    df_train_flipped = generate_proxy_columns(df_train.copy(), protected_columns, noise_param=noise_param)

    if group == "sex":
        Z_train = np.array([0 if z == "Female" else 1 for z in df_train_flipped[f"PROXY_{noise_param:.2f}_sex"]], dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_writter = SummaryWritter(X_train, Y_train, Z_train)
    test_writter = SummaryWritter(X_test, Y_test, Z_test)

    X_train_list = train_writter.grouplabelsplitter(X_train, Y_train, Z_train)
    X_test_list = test_writter.grouplabelsplitter(X_test, Y_test, Z_test)

    return (X_train_list, train_writter), (X_test_list, test_writter)

# (X_train, train_writter), (X_test, test_writter), phats = load_adult_data('../../dataset/adult/processed_data.csv')
# print(f'Type of X_train_list: {type(X_train)}')
# print(phats.shape)
# X_train_numpy = np.concatenate(X_train, axis=0)  # Concatenate along the appropriate axis
# X_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)


# print(phats.shape)
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# Y_train_tensor = torch.tensor(train_writter, dtype=torch.float32).view(-1, 1)  # Reshape for binary output
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# Y_test_tensor = torch.tensor(test_writter, dtype=torch.float32).view(-1, 1)
    
# print(X_train.shape)


