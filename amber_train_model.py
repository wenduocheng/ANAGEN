#%tensorflow_version 1.x
#!python -m pip install -q amber-automl
#!python -m pip install -q keras==2.2.5
import tensorflow as tf
#! if [ ! -d deepsea_train ]; then wget http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz; tar -xvzf deepsea_train_bundle.v0.9.tar.gz; rm deepsea_train_bundle.v0.9.tar.gz; else echo "Found previous downloaded data."; fi



try:
  import amber
  print('AMBER imported successfully')
except ModuleNotFoundError as e:
  print('You need to restart your colab runtime for AMBER to take effect.')
  print('Go to Runtime-->Restart Runtime and run all')
  raise e


# Import everything we will need at top
import os
import shutil
# If you have trouble at this line AFTER installing AMBER, try
# restarting your colab runtime.
try:
  from amber import Amber
  from amber.architect import ModelSpace, Operation
except ModuleNotFoundError:
  print('Restart your Colab runtime by Runtime->restart runtime, and run this cell again')


def get_model_space(out_filters=64, num_layers=9, num_pool=3):
    model_space = ModelSpace()
    expand_layers = [num_layers//num_pool*i-1 for i in range(num_pool)]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


# define data loaders
import h5py
from scipy.io import loadmat
import numpy as np
import pandas as pd


def read_train_data(fp=None):
    fp = fp or "./deepsea_train/train.mat"
    f = h5py.File(fp, "r")
    print(list(f.keys()))
    y = f['traindata'].value
    x = f['trainxdata'].value
    x = np.moveaxis(x, -1, 0)
    y = np.moveaxis(y, -1, 0)
    return x, y


def read_val_data(fp=None):
    fp = fp or "./deepsea_train/valid.mat"
    f = loadmat(fp)
    print(list(f.keys()))
    x = f['validxdata']
    y = f['validdata']
    x = np.moveaxis(x, 1, -1)
    return x, y


def read_test_data(fp=None):
    fp = fp or "./deepsea_train/test.mat"
    f = loadmat(fp)
    print(list(f.keys()))
    x = f['testxdata']
    y = f['testdata']
    x = np.moveaxis(x, 1, -1)
    return x, y

# load the data - will take a while
# Not enough RAM in the free-tier Google Colab
#train_data = read_train_data()

val_data = read_val_data()
# As poor man's version, use test data as train data. For the purpose
# of running AMBER, it will go through; however, the user should use the actuall
# train data, if resouces permit.
train_data = read_test_data()


# First, define the components we need to use
type_dict = {
    'controller_type': 'GeneralController',
    'modeler_type': 'EnasCnnModelBuilder',
    'knowledge_fn_type': 'zero',
    'reward_fn_type': 'LossAucReward',
    'manager_type': 'EnasManager',
    'env_type': 'EnasTrainEnv'
}


# Next, define the specifics
wd = "./outputs/AmberDeepSea/"
if os.path.isdir(wd):
  shutil.rmtree(wd)
os.makedirs(wd)
input_node = Operation('input', shape=(1000, 4), name="input")
output_node = Operation('dense', units=919, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
}

model_space = get_model_space(out_filters=8, num_layers=6)

specs = {
    'model_space': model_space,

    'controller': {
            'share_embedding': {i:0 for i in range(1, len(model_space))},
            'with_skip_connection': True,
            'skip_weight': 1.0,
            'skip_target': 0.4,
            'kl_threshold': 0.01,
            'train_pi_iter': 10,
            'buffer_size': 1,
            'batch_size': 20
    },

    'model_builder': {
        'dag_func': 'EnasConv1dDAG',
        'batch_size': 512,
        'inputs_op': [input_node],
        'outputs_op': [output_node],
        'model_compile_dict': model_compile_dict,
         'dag_kwargs': {
            'stem_config': {
                'flatten_op': 'flatten',
                'fc_units': 100
            }
        }
    },

    'knowledge_fn': {'data': None, 'params': {}},

    'reward_fn': {'method': 'auc'},

    'manager': {
        'data': {
            'train_data': train_data,
            'validation_data': val_data
        },
        'params': {
            'epochs': 1,
            'child_batchsize': 512,
            'store_fn': 'minimal',
            'working_dir': wd,
            'verbose': 2
        }
    },

    'train_env': {
        'max_episode': 20,            # has been reduced for running in colab
        'max_step_per_ep': 20,       # has been reduced for running in colab
        'working_dir': wd,
        'time_budget': "00:15:00",    # has been reduced for running in colab
        'with_input_blocks': False,
        'with_skip_connection': True,
        'child_train_steps': 20,      # has been reduced for running in colab
    }
}



# finally, run program
amb = Amber(types=type_dict, specs=specs)
# you can do some checking/debugging before run
# for example, this returns the controller instance:
amb.controller

amb.run()


