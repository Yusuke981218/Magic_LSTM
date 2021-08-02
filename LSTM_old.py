import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# データ数
datas = 300
# カードの枚数
card_num = 54
# カードの情報
card_info = 9
# マジックの最大段階数
max_flow = 10
# カードの枚数 * カードの情報
card_all = card_num * card_info
# 一時保存リスト
trainX_list, testX_list = [], []
# testの個数
test_cnt = 0

# データ読み込み
for N in range(1, datas + 1):
    df = pd.read_csv('./datasets/' + str(N) + '.csv',
                     index_col=None, header=None, engine='python')

    df_columns = len(df.columns)

    dataset = df.values
    dataset = dataset.astype('float32')

    # データセットを正規化する
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    dan_cnt = 0

    while 8*(dan_cnt+1) <= df_columns:
        dankai_array = dataset[:card_num, 9 * dan_cnt: 9 * (dan_cnt + 1)]
        temp_list = (list(itertools.chain.from_iterable(dankai_array)))
        if len(temp_list) != 0:
            if (N % 10) != 1:
                trainX_list.append(temp_list)
            else:
                test_cnt += 1
                testX_list.append(temp_list)
        dan_cnt += 1

trainX = np.array(trainX_list)
testX = np.array(testX_list)

trainX = np.reshape(trainX, (datas - test_cnt, max_flow, card_all))
trainX = np.reshape(trainX, (test_cnt, max_flow, card_all))

# dataframe = pandas.read_csv('c:/dev/dl/tokyo-weather-2003-2012.csv', usecols=[0,3,4,5,6], engine='python', skipfooter=1)
# plt.plot(dataframe)
# plt.show()
# print(dataframe.head())

# dataset = dataframe.values
# dataset = dataset.astype('float32')

# # データセットを正規化する
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# # トレーニングとテストに分ける
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print(len(train), len(test))

# 値の配列をデータセット行列に変換する
# look_back 3を与えると、配列の一部は以下のようになります。1月、2月、3月


# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         xset = []
#         for j in range(dataset.shape[1]):
#             a = dataset[i:(i+look_back), j]
#             xset.append(a)
#         dataY.append(dataset[i + look_back, 0])
#         dataX.append(xset)
#     return np.array(dataX), np.array(dataY)


# # X=t と Y=t+1 にリシェイプ
# look_back = 12
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# print(testX.shape)
# print(testX[0])
# print(testY)

# 入力を[samples, time steps(変数数), features]にリシェイプする *時系列を列に変換する
# trainX = np.reshape(
#     trainX, (datas, max_flow, card_all)
# testX=np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(max_flow, card_all)))  # shape：変数数、遡る時間数
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=500, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
pad_col = np.zeros(dataset.shape[1]-1)

# invert predictions


def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])


trainPredict = scaler.inverse_transform(pad_array(trainPredict))
trainY = scaler.inverse_transform(pad_array(trainY))
testPredict = scaler.inverse_transform(pad_array(testPredict))
testY = scaler.inverse_transform(pad_array(testY))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# print(testY[:, 0])
# print(testPredict[:, 0])
# # shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(look_back*2) +
#                 1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
