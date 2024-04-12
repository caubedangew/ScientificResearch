import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import time

# Tạo bộ dữ liệu
def preprocessing_data(dataframe):
    dataframe = dataframe.dropna()

    dataframe = dataframe[dataframe["City"] == "Ho Chi Minh City"]

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], format="%d/%m/%Y")

    time_series = np.unique(dataframe["Date"])

    category = ["temperature", "dew", "humidity", "pressure", "wind speed", "pm25"]

    input_layer = np.zeros(shape=(len(time_series), len(category)))

    for x in dataframe.values:
        input_layer[list(time_series).index(x[0])][category.index(x[3]) if category.__contains__(x[3]) else 4] = float(
            x[7])

    input_layer = input_layer[~np.any(input_layer == 0, axis=1)]

    scaler = MinMaxScaler()
    input_layer = scaler.fit_transform(input_layer)

    return pd.DataFrame(data=np.array(input_layer), columns=category)


def prepare_input_for_testing_and_training(X, y):
    # Create a KFold object
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    index = 0

    # Define empty lists to store training and testing data
    X_train_folds, X_test_folds, y_train_folds, y_test_folds = [], [], [], []

    # Loop through each fold
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = dataframe.iloc[train_index, :-1], dataframe.iloc[test_index, :-1]
        y_train, y_test = dataframe.iloc[train_index, -1], dataframe.iloc[test_index, -1]

        # Append the data for this fold to the respective lists
        X_train_folds.append(X_train)
        X_test_folds.append(X_test)
        y_train_folds.append(y_train)
        y_test_folds.append(y_test)

        index += 1

    return X_train_folds, X_test_folds, y_train_folds, y_test_folds


def model(X):
    return tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu',
                                       input_shape=(X.shape[1], 1)),
                tf.keras.layers.MaxPool1D(pool_size=1),
                tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation="relu"),
                tf.keras.layers.MaxPool1D(pool_size=1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(units=1)
            ])


def plot_model_history(model_history):
    plt.plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    plt.plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.show()


if __name__ == '__main__':
    path = r'Dataset/aqi_airqualitydata_2020_en.csv'
    dataframe = pd.read_csv(path)

    dataframe = dataframe[dataframe["City"] == "Ho Chi Minh City"]

    dataframe["Date"] = pd.to_datetime(dataframe["Date"], format="%d/%m/%Y")

    time_series = np.unique(dataframe["Date"])

    category = ["temperature", "dew", "humidity", "pressure", "wind speed", "pm25"]

    input_layer = np.zeros(shape=(len(time_series), len(category)))

    for x in dataframe.values:
        input_layer[list(time_series).index(x[0])][category.index(x[3]) if category.__contains__(x[3]) else 4] = float(
            x[7])

    for i in range(len(category)):
        mean_c, std_c = np.mean(input_layer[:, i]), np.std(input_layer[:, i])
        limit_std = std_c * 3
        lower_std, upper_std = mean_c - limit_std, mean_c + limit_std
        for j in range(len(input_layer[:, i])):
            if input_layer[j, i] <= lower_std or input_layer[j, i] >= upper_std:
                input_layer[j, i] = 0

    input_layer = input_layer[~np.any(input_layer == 0, axis=1)]

    scaler = MinMaxScaler()
    input_layer = scaler.fit_transform(input_layer)

    dataframe = pd.DataFrame(data=np.array(input_layer), columns=category)

    X = dataframe.values[:, :-1]
    y = dataframe.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    X_train_folds, X_test_folds, y_train_folds, y_test_folds = prepare_input_for_testing_and_training(X_train, y_train)

    CNNmodel = model(X_train_folds[0])
    CNNmodel.compile(optimizer="adam", loss=["mae"])

    for i in range(len(X_train_folds)):
        X_train_folds[i] = np.reshape(X_train_folds[i], (X_train_folds[i].shape[0],
                                     X_train_folds[i].shape[1], 1))
        history = CNNmodel.fit(X_train_folds[i], y_train_folds[i], epochs=20, batch_size=32)

    start = time.time()
    CNNmodel.evaluate(X_test, y_test)
    end = time.time()

    print(end - start)


    #
    # test_output = CNNmodel.predict(X_test, batch_size=32)
    # print('Test rmse: ', mean_squared_error(y_test, test_output))
    # forecast_copies = np.repeat(np.reshape(test_output, (test_output.shape[0], 1)), dataframe.values.shape[1], axis=-1)
    # y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
    # forecast_copies = np.repeat(np.reshape(y_test, (y_test.shape[0], 1)), dataframe.values.shape[1], axis=-1)
    # y_test = scaler.inverse_transform(forecast_copies)[:, 0]
    #
    # plt.plot(y_pred_future)
    # plt.plot(y_test)
    # plt.legend(["Testing", "Validate"], loc="upper right")
    # plt.show()
