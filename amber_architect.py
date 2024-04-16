import os
from amber.architect import ModelSpace, Operation
wd = "./outputs/AmberDeepSea/"
input_node = Operation('input', shape=(1000, 4), name="input")
output_node = Operation('dense', units=919, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
}
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
model_space = get_model_space(out_filters=8, num_layers=6)
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



from amber.utils.io import read_history
hist = read_history([os.path.join(wd, "train_history.csv")], 
                    metric_name_dict={'zero':0, 'auc': 1})
hist = hist.sort_values(by='auc', ascending=False)
hist.head(n=5)

from amber.modeler.resnet import ResidualCnnBuilder

keras_builder = ResidualCnnBuilder(
    inputs_op=input_node,
    output_op=output_node,
    fc_units=100,
    flatten_mode='Flatten',
    model_compile_dict={
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
        },
    model_space=model_space,
    dropout_rate=0.1,
    wsf=2
)

best_arc = hist.iloc[0][[x for x in hist.columns if x.startswith('L')]].tolist()
searched_mod = keras_builder(best_arc)


searched_mod.summary()



searched_mod.fit(
    train_data[0],
    train_data[1],
    batch_size=1024,
    validation_data=val_data,
    epochs=5,
    verbose=1
)


from tensorflow.keras.utils import plot_model
from IPython.display import Image

plot_model(searched_mod, show_layer_names=True,
           to_file=os.path.join(wd, "model_arc.png")
           )
Image(filename=os.path.join(wd, 'model_arc.png'), height=1000)
