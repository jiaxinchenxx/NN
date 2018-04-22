import numpy as np

from fcnet import FullyConnectedNet
from utils.solver import Solver
from utils.data_utils import get_CIFAR10_data
import csv

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50%
accuracy on the validation set.
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

DIC = get_CIFAR10_data()



data = {
  'X_train': DIC['X_train'],
  'y_train': DIC['y_train'],
  'X_val': DIC['X_val'],
  'y_val': DIC['y_val'],
}

# configuration of CIFAR-10 training
model = FullyConnectedNet([150,60], num_classes = 10, dropout = 0, reg = 0.01,
                weight_scale = 1e-2, dtype = np.float32, seed = None)


solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={
                  'learning_rate': 0.0003,
                    'momentum' : 0.8
                },
                lr_decay = 0.999,
                num_epochs=20, batch_size=128,
                print_every=100)

solver.train()

with open('raw_data_fcnet.csv', 'w', newline= '') as csvfile:
    fieldname = ['train_acc', 'val_acc', 'train_loss', 'validation_loss']
    writer = csv.DictWriter(csvfile, fieldnames= fieldname)

    for i in range(len(solver.train_acc_history)):
        writer.writerow({'train_acc' : solver.train_acc_history[i], 'val_acc' : solver.val_acc_history[i]})

    for i in range(len(solver.loss_history)):
        writer.writerow({'train_loss' : solver.loss_history[i], 'validation_loss' : solver.loss_history_val[i]})

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
