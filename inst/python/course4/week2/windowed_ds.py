import tensorflow as tf


def windowed_ds(series, window_size, batch_size, shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size+1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size+1))
    ds = ds.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds


"""
window_size = 20
batch_size = 32
shuffle_buff_size = 1000

ds = windowed_dataset(list(range(100)), window_size, batch_size, shuffle_buff_size)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(ds, epochs=100, verbose=0)


print("Layer weights {}".format(l0.get_weights())

# Simple linear regression
import numpy as np
print(series[1:21])
print(model.predict(series[1:21][np.newaxis]))


forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis])))

forecast= forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


tf.keras.metrics.mean_absolute_error(x_valid, results)
"""
