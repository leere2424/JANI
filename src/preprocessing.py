import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# train data용 preprocess 함수
# LabelEncoder와 StandardScaler를 이용해서 fit_transform해주는 함수
def fit_preprocessing(data):

    '''
    LabelEncoder와 StandardScaler를 이용해서 fit_transform해주는 함수
    :return: data_scaled, encoders, scaler
    '''

    data = data.copy()

    features = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']

    # encoding
    encoders = {}

    for feature in features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
        encoders[feature] = encoder

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    data_scaled = pd.DataFrame(
        data_scaled,
        columns=data.columns,
        index=data.index
    )

    return data_scaled, encoders, scaler


# val data / test data용 preprocess 함수
# train data에서 fit한 LabelEncoder와 StandardScaler를 이용해 transform해주는 함수
def transform_preprocessor(data, encoders, scaler):

    '''
    train data에서 fit한 LabelEncoder와 StandardScaler를 이용해 transform해주는 함수
    :return: data_scaled
    '''

    data = data.copy()

    features = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']

    for feature in features:
        encoder = encoders[feature]
        data[feature] = encoder.transform(data[feature])

    data_scaled = scaler.transform(data)

    data_scaled = pd.DataFrame(
        data_scaled,
        columns=data.columns,
        index=data.index
    )

    return data_scaled