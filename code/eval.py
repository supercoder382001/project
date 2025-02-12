import tensorflow as tf
import configparser
from read_dataset import input_fn
from routenet_model_occu_eval_1 import RouteNetModelOccuEval_1
from routenet_model_occu_eval_2 import RouteNetModelOccuEval_2
import numpy as np

# Ensure GPU is used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def trnsf_minmax(x, y, feats):
    for k, v in feats.items():
        denom = v[1] - v[0]
        x[k] = tf.math.divide(tf.math.subtract(x[k], v[0]), denom)
    return x, y

st_minmax = {
    'traffic': [30.787, 2048.23],
    'packets': [0.032895, 2.03633],
    'capacity': [10000, 100000],
    'eqlambda': [40.0337, 1999.52]
}

# Read config
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Optimized Dataset Loading
ds_test = input_fn(config['DIRECTORIES']['test'], shuffle=False)
ds_test = ds_test.map(lambda x, y: trnsf_minmax(x, y, st_minmax), num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Load models
model1 = RouteNetModelOccuEval_1(config)
model2 = RouteNetModelOccuEval_2(config)

dir_model1 = 'saved_best_model_1/model1'
dir_model2 = 'saved_best_model_2/model2'

model1.load_weights(dir_model1)
model2.load_weights(dir_model2)

# Use tf.function and direct model call for faster predictions
# Predictions without @tf.function to avoid tensor access issues
print('Starting evaluation...')
pred1 = model1.predict(ds_test, verbose=0)
print('Prediction model 1 Done')

pred2 = model2.predict(ds_test, verbose=0)
print('Prediction model 2 Done')

# Ensemble Predictions
pred = tf.concat([pred1, pred2], axis=1)
predictions = tf.reduce_mean(pred, axis=1, keepdims=True)
print('Prediction ensemble Done')

# Collect Targets
target = tf.concat([y for _, y in ds_test], axis=0)

# Evaluate MAPE
loss_object = tf.keras.losses.MeanAbsolutePercentageError()
MAPE_ensemble = loss_object(np.squeeze(target), np.squeeze(predictions))

print('MAPE ensemble =', MAPE_ensemble.numpy())
