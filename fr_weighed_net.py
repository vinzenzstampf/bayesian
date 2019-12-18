'''
Resources:
https://en.wikipedia.org/wiki/Universal_approximation_theorem
http://neuralnetworksanddeeplearning.com/chap4.html
https://github.com/thomberg1/UniversalFunctionApproximation
https://cms-nanoaod-integration.web.cern.ch/integration/master-102X/mc102X_doc.html
https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
'''

import root_pandas

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from root_numpy import root2array

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from keras import backend as K
from keras.activations import softmax
from keras.constraints import unit_norm
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from pdb import set_trace
       
# fix random seed for reproducibility (FIXME! not really used by Keras)
np.random.seed(1986)

# define input features
features = [
    'ele_pt',
#     'abs(ele_eta)',
#     'abs(ele_dxy)',
#     'abs(ele_dz)',
#     'ele_eta',
#     'ele_dxy',
#     'ele_dz',
    'ele_phi',
#     'ele_iso',
#     'ngvtx',
    'rho',
]

branches = features + [
    'ele_genPartFlav',
    'ele_iso',
    'ele_id',
    'z_eta',
    'z_pt',
    'z_mass',
    'z_phi',

    'ele_pt',
    'ele_eta',
    'ele_dxy',
    'ele_dz',
    'ele_phi',

    'ngvtx',
    'rho',
]

branches = list(set(branches))

filein = 'dy1j_v3.root'

# load dataset including all event, both passing and failing
passing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection='ele_iso<20 & ele_id>0.5 & ele_iso<0.15 & abs(ele_dz)<0.2 & abs(ele_dxy)<0.05') )
failing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection='ele_iso<20 & ele_id>0.5 & ele_iso>0.15 & abs(ele_dz)<0.2 & abs(ele_dxy)<0.05') )

# passing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection='ele_iso<2 &   ele_iso<0.15 & ele_id>0.5 ' ) )
# failing = pd.DataFrame( root2array(filein, 'tree', branches=branches, selection='ele_iso<2 & !(ele_iso<0.15 & ele_id>0.5)') )


# targets
passing['target'] = np.ones (passing.shape[0]).astype(np.int)
failing['target'] = np.zeros(failing.shape[0]).astype(np.int)

# concatenate the events and shuffle
data = pd.concat([passing, failing])
# add abs eta, dxy, dz
data['abs_ele_dxy'] = np.abs(data.ele_dxy)
data['abs_ele_dz' ] = np.abs(data.ele_dz )
data['abs_ele_eta'] = np.abs(data.ele_eta)
features += [
    'abs_ele_dxy',
    'abs_ele_dz' ,
    'abs_ele_eta',
]

# reindex to avoid duplicated indices, useful for batches
# https://stackoverflow.com/questions/27236275/what-does-valueerror-cannot-reindex-from-a-duplicate-axis)%20-mean
data.index = np.array(range(len(data)))
data = data.sample(frac=1, replace=False, random_state=1986) # shuffle

# subtract genuine electrons using negative weights
data['isnonprompt'] =   1. * (data['ele_genPartFlav'] == 0)  \
                      + 1. * (data['ele_genPartFlav'] == 3)  \
                      + 1. * (data['ele_genPartFlav'] == 4)  \
                      + 1. * (data['ele_genPartFlav'] == 5)  \
                      - 1. * (data['ele_genPartFlav'] == 1)  \
                      - 1. * (data['ele_genPartFlav'] == 15) \
                      - 1. * (data['ele_genPartFlav'] == 22)

# X and Y
X = pd.DataFrame(data, columns=list(set(branches+features+['isnonprompt'])))
Y = pd.DataFrame(data, columns=['target'])

# activation = 'tanh'
activation = 'selu'
# activation = 'sigmoid'
# activation = 'relu'
# activation = 'LeakyReLU' #??????

# define the net
input  = Input((len(features),))
layer  = Dense(1024, activation=activation   , name='dense1', kernel_constraint=unit_norm())(input)
# layer  = Dropout(0.8, name='dropout1')(layer)
# layer  = BatchNormalization()(layer)
# layer  = Dense(256, activation=activation   , name='dense2', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.8, name='dropout2')(layer)
# layer  = BatchNormalization()(layer)
# layer  = Dense(256, activation=activation   , name='dense3', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.8, name='dropout3')(layer)
# layer  = BatchNormalization()(layer)
layer  = Dense(64, activation=activation   , name='dense4', kernel_constraint=unit_norm())(layer)
layer  = Dropout(0.8, name='dropout4')(layer)
# layer  = BatchNormalization()(layer)
# layer  = Dense(16, activation=activation   , name='dense5', kernel_constraint=unit_norm())(layer)
# layer  = Dropout(0.5, name='dropout5')(layer)
# layer  = BatchNormalization()(layer)
output = Dense(  1, activation='sigmoid', name='output', )(layer)

# Define outputs of your model
model = Model(input, output)

# choose your optimizer
# opt = SGD(lr=0.0001, momentum=0.8)
opt = 'Adam'

# compile and choose your loss function (binary cross entropy for a 1-0 classification problem)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae', 'acc'])

# print net summary
print( model.summary() )

# plot the models
# https://keras.io/visualization/
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


# normalize inputs FIXME! do it, but do it wisely
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# on QuantileTransformer
# Note that this transform is non-linear. It may distort linear
#     correlations between variables measured at the same scale but renders
#     variables measured at different scales more directly comparable.
# from sklearn.preprocessing import QuantileTransformer
# qt = QuantileTransformer(output_distribution='normal', random_state=1986)
# fit and FREEZE the transformation paramaters. 
# Need to save these to be applied exactly as are when predicting on a different dataset
# qt.fit(X[features])
# now transform
# xx = qt.transform(X[features])

# alternative way to scale the inputs
# https://datascienceplus.com/keras-regression-based-neural-networks/

from sklearn.preprocessing import RobustScaler
qt = RobustScaler()
qt.fit(X[features])
xx = qt.transform(X[features])

# save the frozen transformer
pickle.dump( qt, open( 'input_tranformation_weighted.pck', 'wb' ) )

# early stopping
# monitor = 'val_acc'
monitor = 'val_loss'
# monitor = 'val_mean_absolute_error'
es = EarlyStopping(monitor=monitor, mode='auto', verbose=1, patience=100, restore_best_weights=True)

# reduce learning rate when at plateau, fine search the minimum
reduce_lr = ReduceLROnPlateau(monitor=monitor, mode='auto', factor=0.5, patience=5, min_lr=0.001, cooldown=20, verbose=True)

# train only the classifier. beta is set at 0 and the discriminator is not trained
# history = model.fit(xx, Y, epochs=2000, validation_split=0.5, callbacks=[es, reduce_lr], batch_size=32, sample_weight=np.array(X['isnonprompt']))  
history = model.fit(xx, Y, epochs=2000, validation_split=0.5, callbacks=[es, reduce_lr], batch_size=32)  # FIXME # DEFAULT 
# history = model.fit(xx, Y, epochs=2, validation_split=0.5, callbacks=[es, reduce_lr], batch_size=32)  #TEST 

# plot loss function trends for train and validation sample
plt.clf()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plt.yscale('log')
plt.ylim((0.48,0.5))
plt.grid(True)
plt.savefig('loss_function_history_weighted.pdf')
plt.clf()

# plot accuracy trends for train and validation sample
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.yscale('log')
plt.savefig('accuracy_history_weighted.pdf')
plt.clf()

# plot accuracy trends for train and validation sample
plt.plot(history.history['mae'], label='train')
plt.plot(history.history['val_mae'], label='test')
plt.legend()
plt.yscale('log')
plt.savefig('mean_absolute_error_history_weighted.pdf')
plt.clf()

# calculate predictions on the data sample
print ('predicting on', data.shape[0], 'events')
x = pd.DataFrame(data, columns=features)
# y = model.predict(x)
# load the transformation with the correct parameters!
qt = pickle.load(open( 'input_tranformation_weighted.pck', 'rb' ))
xx = qt.transform(x[features])
y = model.predict(xx)

# impose norm conservation if you want probabilities
# compute the overall rescaling factor scale
scale = 1.
# scale = np.sum(passing['target']) / np.sum(y)

# add the score to the data sample
data.insert(len(data.columns), 'weight', scale * y)

# let sklearn do the heavy lifting and compute the ROC curves for you
fpr, tpr, wps = roc_curve(data.target, data.weight) 
plt.plot(fpr, tpr)
plt.yscale('linear')
plt.savefig('roc_weighted.pdf')

# save model and weights
model.save('net_model_weighted.h5')
# model.save_weights('net_model_weights.h5')

# rename branches, if you want
# data.rename(
#     index=str, 
#     columns={'cand_refit_mass12': 'mass12',}, 
#     inplace=True)

# save ntuple
data.to_root('output_ntuple_weighted.root', key='tree', store_index=False)

