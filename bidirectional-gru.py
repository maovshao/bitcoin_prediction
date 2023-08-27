# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
print(pd.__version__)

# load the dataset and show a part it
df = pd.read_csv('sentiment-bitcoin.csv')
df = df.rename(columns = {'Unnamed: 0': 'timestamp'})
print(df.head())

# %% [markdown]
# ## Simple metrics study

# print describsion of Polarity
print(f"describsion of Polarity: \n {df['Polarity'].describe()}")

# print describsion of Sensitivity
print(f"describsion of Sensitivity: \n {df['Sensitivity'].describe()}")

# print describsion of Tweet_vol
print(f"describsion of Tweet_vol: \n {df['Tweet_vol'].describe()}")

# print describsion of Close_Price
print(f"describsion of Close_Price: \n {df['Close_Price'].describe()}")

# %% [markdown]
# ## Detecting outliers / sudden spikes in our close prices

# %%
def detect(signal, treshold = 2.0):
    detected = []
    for i in range(len(signal)):
        if np.abs(signal[i]) > treshold:
            detected.append(i)
    return detected

# %%
signal = np.copy(df['Close_Price'].values)
std_signal = (signal - np.mean(signal)) / np.std(signal)
s = pd.Series(std_signal)
s.describe(percentiles = [0.25, 0.5, 0.75, 0.95])

# %%
outliers = detect(std_signal, 1.3)

# %%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), signal)
plt.plot(
    np.arange(len(signal)),
    signal,
    'X',
    label = 'outliers',
    markevery = outliers,
    c = 'r',
)
plt.xticks(
    np.arange(len(signal))[::15], df['timestamp'][::15], rotation = 'vertical'
)
plt.savefig("./output_figure/outliers.png")
plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler().fit(df[['Polarity', 'Sensitivity', 'Close_Price']])
scaled = minmax.transform(df[['Polarity', 'Sensitivity', 'Close_Price']])

# %%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), scaled[:, 0], label = 'Scaled polarity')
plt.plot(np.arange(len(signal)), scaled[:, 1], label = 'Scaled sensitivity')
plt.plot(np.arange(len(signal)), scaled[:, 2], label = 'Scaled closed price')
plt.plot(
    np.arange(len(signal)),
    scaled[:, 0],
    'X',
    label = 'outliers polarity based on closed price',
    markevery = outliers,
    c = 'r',
)
plt.plot(
    np.arange(len(signal)),
    scaled[:, 1],
    'o',
    label = 'outliers sensitivity based on closed price',
    markevery = outliers,
    c = 'r',
)
plt.xticks(
    np.arange(len(signal))[::15], df['timestamp'][::15], rotation = 'vertical'
)
plt.legend()
plt.savefig("./output_figure/Scaled_data.png")
plt.show()

# %% [markdown]
# Doesnt show much from trending, how about covariance correlation?

# %% [markdown]
# ## Pearson correlation

# %%
colormap = plt.cm.RdBu
plt.figure(figsize = (15, 7))
plt.title('pearson correlation', y = 1.05, size = 16)

mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    df.corr(),
    mask = mask,
    linewidths = 0.1,
    vmax = 1.0,
    square = True,
    cmap = colormap,
    linecolor = 'white',
    annot = True,
)
plt.savefig("./output_figure/Pearson_correlation.png")
plt.show()

# %%
def df_shift(df, lag = 0, start = 1, skip = 1, rejected_columns = []):
    df = df.copy()
    if not lag:
        return df
    cols = {}
    for i in range(start, lag + 1, skip):
        for x in list(df.columns):
            if x not in rejected_columns:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k, v in cols.items():
        columns = v
        dfn = pd.DataFrame(data = None, columns = columns, index = df.index)
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods = i)
            i += 1
        df = pd.concat([df, dfn], axis = 1)
    return df

# %%
df_new = df_shift(df, lag = 42, start = 7, skip = 7)
df_new.shape

# %%
colormap = plt.cm.RdBu
plt.figure(figsize = (30, 20))
ax = plt.subplot(111)
plt.title('42 hours correlation', y = 1.05, size = 16)
selected_column = [
    col
    for col in list(df_new)
    if any([k in col for k in ['Polarity', 'Sensitivity', 'Close']])
]

sns.heatmap(
    df_new[selected_column].corr(),
    ax = ax,
    linewidths = 0.1,
    vmax = 1.0,
    square = True,
    cmap = colormap,
    linecolor = 'white',
    annot = True,
)
plt.savefig("./output_figure/42_hours_correlation.png")
plt.show()

# %% [markdown]
# ## How about we check trends from moving average? i chose 7, 14, 30 hours

# %% [markdown]
# I think i had too much playing daily trending data

# %%
def moving_average(signal, period):
    buffer = [np.nan] * period
    for i in range(period, len(signal)):
        buffer.append(signal[i - period : i].mean())
    return buffer

# %%
signal = np.copy(df['Close_Price'].values)
ma_7 = moving_average(signal, 7)
ma_14 = moving_average(signal, 14)
ma_30 = moving_average(signal, 30)

# %%
plt.figure(figsize = (15, 7))
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.plot(np.arange(len(signal)), ma_7, label = 'ma 7')
plt.plot(np.arange(len(signal)), ma_14, label = 'ma 14')
plt.plot(np.arange(len(signal)), ma_30, label = 'ma 30')
plt.legend()
plt.savefig("./output_figure/trends_from_moving_average.png")
plt.show()

# %% [markdown]
# Trends gonna increase anyway!

# %% [markdown]
# ## Now deep learning LSTM

# %%
num_layers = 1
learning_rate = 0.005
size_layer = 128
timestamp = 5
epoch = 200
dropout_rate = 0.6

# %%
dates = pd.to_datetime(df.iloc[:, 0]).tolist()

# %%
class Model:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        forget_bias = 0.1,
    ):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.GRUCell(size_layer)

        backward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        forward_rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, size))
        drop_backward = tf.contrib.rnn.DropoutWrapper(
            backward_rnn_cells, output_keep_prob = forget_bias
        )
        forward_backward = tf.contrib.rnn.DropoutWrapper(
            forward_rnn_cells, output_keep_prob = forget_bias
        )
        self.backward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * size_layer)
        )
        self.forward_hidden_layer = tf.placeholder(
            tf.float32, shape = (None, num_layers * size_layer)
        )
        self.outputs, self.last_state = tf.nn.bidirectional_dynamic_rnn(
            forward_backward,
            drop_backward,
            self.X,
            initial_state_fw = self.forward_hidden_layer,
            initial_state_bw = self.backward_hidden_layer,
            dtype = tf.float32,
        )
        self.outputs = tf.concat(self.outputs, 2)
        self.logits = tf.layers.dense(self.outputs[-1], size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100

def calculate_accuracy_scaled(predict, real):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    from sklearn.metrics import r2_score
    return r2_score(real,predict)
# %%
minmax = MinMaxScaler().fit(
    df[['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']].astype(
        'float32'
    )
)
df_scaled = minmax.transform(
    df[['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']].astype(
        'float32'
    )
)
df_scaled = pd.DataFrame(df_scaled)
df_scaled.head()

# %%
tf.compat.v1.reset_default_graph()
modelnn = Model(
    learning_rate, num_layers, df_scaled.shape[1], size_layer, dropout_rate
)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
import tensorflow.contrib.slim as slim
model_vars = tf.trainable_variables()
slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# %% [markdown]
# We need to scale our data between 0 - 1 or any scaled you wanted, but must not less than -1 and more than 1, because LSTM is using tanh function, squashing high values can caused gradient vanishing later
Loss_history = []
# %%
for i in range(epoch):
    init_value_forward = np.zeros((1, num_layers * size_layer))
    init_value_backward = np.zeros((1, num_layers * size_layer))
    total_loss = 0
    for k in range(0, (df_scaled.shape[0] // timestamp) * timestamp, timestamp):
        batch_x = np.expand_dims(
            df_scaled.iloc[k : k + timestamp].values, axis = 0
        )
        batch_y = df_scaled.iloc[k + 1 : k + timestamp + 1].values
        last_state, _, loss = sess.run(
            [modelnn.last_state, modelnn.optimizer, modelnn.cost],
            feed_dict = {
                modelnn.X: batch_x,
                modelnn.Y: batch_y,
                modelnn.backward_hidden_layer: init_value_backward,
                modelnn.forward_hidden_layer: init_value_forward,
            },
        )
        init_value_forward = last_state[0]
        init_value_backward = last_state[1]
        total_loss += loss
    total_loss /= df.shape[0] // timestamp
    Loss_history.append(total_loss)
    if (i + 1) % 20 == 0:
        print('epoch:', i + 1, 'avg loss:', total_loss)

plt.figure(figsize = (15, 7))

plt.plot(np.arange(len(Loss_history)), Loss_history, label = 'Loss')
plt.legend()
plt.savefig("./output_figure/BI-GRU_Loss.png")
plt.show()

# %%
def predict_future(future_count, df, dates, indices = {}):
    df = df[0:-future_count]
    dates = dates[0:-future_count]
    date_ori = dates[:]
    cp_df = df.copy()
    output_predict = np.zeros((cp_df.shape[0] + future_count, cp_df.shape[1]))
    output_predict[0] = cp_df.iloc[0]
    upper_b = (cp_df.shape[0] // timestamp) * timestamp
    init_value_forward = np.zeros((1, num_layers * size_layer))
    init_value_backward = np.zeros((1, num_layers * size_layer))
    for k in range(0, (df.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(
                    cp_df.iloc[k : k + timestamp], axis = 0
                ),
                modelnn.backward_hidden_layer: init_value_backward,
                modelnn.forward_hidden_layer: init_value_forward,
            },
        )
        init_value = last_state
        output_predict[k + 1 : k + timestamp + 1] = out_logits
    out_logits, last_state = sess.run(
        [modelnn.logits, modelnn.last_state],
        feed_dict = {
            modelnn.X: np.expand_dims(cp_df.iloc[upper_b:], axis = 0),
            modelnn.backward_hidden_layer: init_value_backward,
            modelnn.forward_hidden_layer: init_value_forward,
        },
    )
    init_value = last_state
    output_predict[upper_b + 1 : cp_df.shape[0] + 1] = out_logits
    cp_df.loc[cp_df.shape[0]] = out_logits[-1]
    date_ori.append(date_ori[-1] + timedelta(hours = 1))
    if indices:
        for key, item in indices.items():
            cp_df.iloc[-1, key] = item
    for i in range(future_count - 1):
        out_logits, last_state = sess.run(
            [modelnn.logits, modelnn.last_state],
            feed_dict = {
                modelnn.X: np.expand_dims(cp_df.iloc[-timestamp:], axis = 0),
                modelnn.backward_hidden_layer: init_value_backward,
                modelnn.forward_hidden_layer: init_value_forward,
            },
        )
        init_value = last_state
        output_predict[cp_df.shape[0]] = out_logits[-1]
        cp_df.loc[cp_df.shape[0]] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(hours = 1))
        if indices:
            for key, item in indices.items():
                cp_df.iloc[-1, key] = item
    return {'date_ori': date_ori, 'df': cp_df.values}

# %% [markdown]
# Define some smoothing, using previous value as an anchor

# %%
def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

# %%
predict_30 = predict_future(30, df_scaled, dates)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])

# %%
plt.figure(figsize = (15, 7))
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label = 'predict signal',
)
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.legend()
plt.savefig("./output_figure/BI-GRU_predict_signal.png")
plt.show()

accuracy_train = calculate_accuracy(anchor(predict_30['df'][0:-30, -1], 0.5), signal[0:-30])
print(f"accuracy_train = {accuracy_train}")
accuracy_test = calculate_accuracy(anchor(predict_30['df'][-30:-1, -1], 0.5), signal[-30:-1])
print(f"accuracy_test = {accuracy_test}")

accuracy_train_scaled = calculate_accuracy_scaled(anchor(predict_30['df'][0:-30, -1], 0.5), signal[0:-30])
print(f"accuracy_train_scaled = {accuracy_train_scaled}")
accuracy_test_scaled = calculate_accuracy_scaled(anchor(predict_30['df'][-30:-1, -1], 0.5), signal[-30:-1])
print(f"accuracy_test_scaled = {accuracy_test_scaled}")

# %% [markdown]
# #### What happen if polarity is double from the max? Polarity is first index

# %%
scaled_polarity = (minmax.data_max_[0] * 2 - minmax.data_min_[0]) / (
    minmax.data_max_[0] - minmax.data_min_[0]
)
scaled_polarity

# %%
plt.figure(figsize = (15, 7))
predict_30 = predict_future(
    30, df_scaled, dates, indices = {0: scaled_polarity}
)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label = 'predict signal',
)
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.legend()
plt.savefig("./output_figure/BI-GRU_predict_signal_polarity_double_max.png")
plt.show()

# %% [markdown]
# I retried for 3 times just to study how fitted our model is, if every retry has big trend changes, so we need to retrain again.

# %% [markdown]
# #### What happen if polarity is quadriple from the min? polarity is first index

# %%
scaled_polarity = (minmax.data_min_[0] / 2 - minmax.data_min_[0]) / (
    minmax.data_max_[0] - minmax.data_min_[0]
)
scaled_polarity

# %%
plt.figure(figsize = (15, 7))

predict_30 = predict_future(
    30, df_scaled, dates, indices = {0: scaled_polarity}
)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label = 'predict signal',
)
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.legend()
plt.savefig("./output_figure/BI-GRU_predict_signal_polarity_double_min.png")
plt.show()

# %% [markdown]
# The second graph is skewed, but we got 2 graphs represented positive trends

# %% [markdown]
# As you can see, the model learnt that, polarity gives negative correlation to the model. If polarity is increase, the trend is decreasing, vice versa

# %% [markdown]
# #### What happen if sentiment volume is double from the max? Volume is third index

# %%
scaled_volume = (minmax.data_max_[2] * 2 - minmax.data_min_[2]) / (
    minmax.data_max_[2] - minmax.data_min_[2]
)
scaled_volume


# %%
plt.figure(figsize = (15, 7))

predict_30 = predict_future(
    30, df_scaled, dates, indices = {2: scaled_volume}
)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label = 'predict signal',
)
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.legend()
plt.savefig("./output_figure/BI-GRU_predict_signal_sentiment_double_max.png")
plt.show()

# %% [markdown]
# #### What happen if sentiment volume is double from the min? Volume is third index

# %%
scaled_volume = (minmax.data_min_[2] / 2 - minmax.data_min_[2]) / (
    minmax.data_max_[2] - minmax.data_min_[2]
)
scaled_volume

# %%
plt.figure(figsize = (15, 7))

predict_30 = predict_future(
    30, df_scaled, dates, indices = {2: scaled_volume}
)
predict_30['df'] = minmax.inverse_transform(predict_30['df'])
plt.plot(
    np.arange(len(predict_30['date_ori'])),
    anchor(predict_30['df'][:, -1], 0.5),
    label = 'predict signal',
)
plt.plot(np.arange(len(signal)), signal, label = 'real signal')
plt.legend()
plt.savefig("./output_figure/BI-GRU_predict_signal_sentiment_double_min.png")
plt.show()

# %% [markdown]
# As you can see, volume does not brings any impact the learning so much

# %%



