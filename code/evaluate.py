#import os

# Uncomment this line in case you want to disable GPU execution
# Note you need to have CUDA installed to run de execution in GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import configparser
from read_dataset import input_fn
from routenet_model_occu_eval_1 import RouteNetModelOccuEval_1
from routenet_model_occu_eval_2 import RouteNetModelOccuEval_2
import numpy as np

def trnsf_minmax(x, y, feats):
    """Apply a transformation over all the samples included in the dataset.
            Args:
                x (dict): predictor variables.
                y (array): target variable.
                feats: dictionary of features with associated max to apply
            Returns:
                x,y: The modified predictor/target variables.
    """

    for k,v in feats.items():
        denom = v[1] - v[0]
        x[k] = tf.math.divide(tf.math.subtract(x[k], v[0]), denom)

    return x, y

st_minmax = {
        'traffic': [30.787, 2048.23],
        'packets': [0.032895, 2.03633],
        'capacity':[10000, 100000],
        'eqlambda':[40.0337, 1999.52]
        }

# Read the config.ini file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

ds_test = input_fn(config['DIRECTORIES']['test'], shuffle=False)
ds_test = ds_test.map(lambda x, y: trnsf_minmax(x, y, st_minmax))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)



# Instantiate the models
model1 = RouteNetModelOccuEval_1(config)
model2 = RouteNetModelOccuEval_2(config)

# path of the best models
# comment these 4 lines in case you want to load the trained neural network we provided
with open('Best_model_1.txt') as f:
    dir_model1 = f.readlines()[0]
with open('Best_model_2.txt') as f:
    dir_model2 = f.readlines()[0]

# Uncomment these two lines in case you want to load the trained neural network we provided
# dir_model1 = 'saved_best_model_1/model1'
# dir_model2 = 'saved_best_model_2/model2'

# Load the models
model1.load_weights(dir_model1)

model2.load_weights(dir_model2)

print('starting evalution')
print('prediction model 1 in progress ...')
pred1 = model1.predict(ds_test, verbose=1)

print('prediction model 1 Done')
print('prediction model 2 in progress ...')
pred2 = model2.predict(ds_test, verbose=1)
print('prediction model 2 Done')

pred = tf.concat([pred1,pred2],axis=1)
predictions = tf.math.reduce_mean(pred, axis=1, keepdims=True)
print('prediction ensemble Done')

print('collecting target in progress ...')
target = tf.concat([y for x, y in ds_test], axis=0)
print('collecting target Done')

print('Finalizing the evaluation ...')

loss_object = tf.keras.losses.MeanAbsolutePercentageError()
MAPE_ensemble = loss_object(np.squeeze(target),np.squeeze(predictions))

print('MAPE ensemble =', MAPE_ensemble.numpy())




