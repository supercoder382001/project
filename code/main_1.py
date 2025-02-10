import tensorflow as tf
import sys
import numpy as np

sys.path.insert(1, "./code")
from read_dataset_occu import input_fn_occu
from routenet_model_occu_1 import RouteNetModelOccu_1
from util_funcs import set_seeds
import configparser
import os


# In case you want to disable GPU execution uncomment this line
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



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

#Dictionary with standardization values

st_minmax = {
        'traffic': [30.787, 2048.23],
        'packets': [0.032895, 2.03633],
        'capacity':[10000, 100000],
        'eqlambda':[40.0337, 1999.52]
        }

#set the random seed
set_seeds(4)

# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

# Initialize the datasets
ds_train = input_fn_occu(config['DIRECTORIES']['train'], shuffle=True)
ds_train = ds_train.map(lambda x, y: trnsf_minmax(x, y, st_minmax))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_test = input_fn_occu(config['DIRECTORIES']['val_3'], shuffle=False)
ds_test = ds_test.map(lambda x, y: trnsf_minmax(x, y, st_minmax))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=float(config['HYPERPARAMETERS']['learning_rate']))

# Define, build and compile the model
model = RouteNetModelOccu_1(config)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics="MAPE")

# Define the checkpoint directory where the model will be saved
ckpt_dir = config['DIRECTORIES']['logs_model1']

latest = tf.train.latest_checkpoint(ckpt_dir)

# Reload the pretrained model in case it exists
if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{MAPE:.2f}-{val_MAPE:.2f}")

# If save_best_only, the program will only save the best model using 'monitor' as metric
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode='min',
    monitor='val_MAPE',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

# This method trains the model saving the model each epoch.
history = model.fit(ds_train,
          epochs=int(config['RUN_CONFIG']['epochs']),
          steps_per_epoch=int(config['RUN_CONFIG']['steps_per_epoch']),
          validation_data=ds_test,
          validation_steps=int(config['RUN_CONFIG']['validation_steps']),
          callbacks=[cp_callback],
          use_multiprocessing=True)
np.save('history_training_model1.npy', history.history)