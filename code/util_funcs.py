import tensorflow as tf
import sys
import numpy as np
import os
import random

sys.path.insert(1, "./code")

def set_seeds(seed):
    """ Set random seed
    
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    return 0

def main():
    #test code here

    a = random.randint(0,100)

    print("No seed:")
    print(a)

    set_seeds(4)

    b = random.randint(0,100)

    print("With seed:")
    print(b)

    return 0

if __name__ == "__main__":
    main()

