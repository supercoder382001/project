import tensorflow as tf
import sys
sys.path.insert(1, "./code")
from read_dataset import input_fn
import pandas as pd
from routenet_model_occu_eval_1 import RouteNetModelOccuEval_1
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


def find_top_k(path, ranges={(0,50):2, (51,100):2}):
    """Find the top k models according to the training.
            Args:
                path (str): location of the saved models.
                ranges (dict): dictionary of epoch ranges with associated number of models we want to select.
                
            Returns:
                The list of filename of the top k models.
    """
    #dataframe to save the result
    df_RES = pd.DataFrame()
    
    epoch = []
    MAPE = []
    name = []
    
    for filename in os.listdir(path):
        if filename.endswith('.index'):
            data = os.path.splitext(filename)[0].split('-')
            epoch.append(int(data[0]))
            MAPE.append(float(data[-1]))
            name.append(filename)
            
    df_RES['epoch'] = epoch
    df_RES['MAPE'] = MAPE
    df_RES['Filename'] = name
    
    result = []
    
    #extraction of best models per range
    [result.extend(list(df_RES[(df_RES['epoch'] >= p[0]) & (df_RES['epoch'] <= p[1])].sort_values(['MAPE'], ascending=True).head(ranges[p])['Filename'].values)) for p in ranges.keys()]
    
    return result


st_minmax = {
        'traffic': [30.787, 2048.23],
        'packets': [0.032895, 2.03633],
        'capacity':[10000, 100000],
        'eqlambda': [40.0337, 1999.52]
        }

# Read the config file
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')



# Initialize the datasets
ds_valf_del = input_fn(config['DIRECTORIES']['test'], shuffle=False)
ds_valf_del = ds_valf_del.map(lambda x, y: trnsf_minmax(x, y, st_minmax))
ds_valf_del = ds_valf_del.prefetch(tf.data.experimental.AUTOTUNE)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=float(config['HYPERPARAMETERS']['learning_rate']))

# Define, build and compile the model
model = RouteNetModelOccuEval_1(config)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics="MAPE")




def main(path, ranges):
    """Find the best models out of the k selected ones from the training.
            Args:
                path (str): location of the saved models.
                ranges (dict): dictionary of epoch ranges with associated number of models we want to select.
                
            Returns:
                The filename of the top k models and save in a txt file.
    """
    #find top k models according to the training val
    top_k_models = find_top_k(path, ranges)
    
    #dict to save the MAPE
    result = {}
    print("Starting find best model")
    
    for filename in top_k_models:
        
        model_dir = path + '/' + os.path.splitext(filename)[0] 
        model.load_weights(model_dir)
        
        MAPE = model.evaluate(ds_valf_del)
        
        result[model_dir] = MAPE[0]
        print('model ', filename, ' MAPE = ', MAPE[0])
        
    Best_model = list(result.keys())[list(result.values()).index(min(result.values()))]
    print('The best model is model ', Best_model)
    
    Best_model_file = open('Best_model_1.txt', "w")
    Best_model_file.write(Best_model)
    Best_model_file.close()
    
    return Best_model


if __name__=="__main__":
    path = config['DIRECTORIES']['logs_model1']
    ranges = {
        (0,100):2,
        (101,200):2
        }
    
    main(path, ranges)
