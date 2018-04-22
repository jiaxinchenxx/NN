import numpy as np

from fcnet import FullyConnectedNet
from utils.solver import Solver
from utils.data_utils import get_CIFAR10_data
import csv

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
###########################################################################
#                           BEGIN OF YOUR CODE                            #
###########################################################################

DIC = get_CIFAR10_data()
data = {
  'X_train': DIC['X_train'][0:49],
  'y_train': DIC['y_train'][0:49],
  'X_val': DIC['X_val'],
  'y_val': DIC['y_val'],
}



# configuration of overfitting
model = FullyConnectedNet([20,30], num_classes = 10, dropout = 0, reg = 0.5,
                weight_scale = 1e-2, dtype = np.float32, seed = 42)


solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 0.0033,
                },
                lr_decay=1,
                num_epochs=20, batch_size=10,
                print_every=100)

solver.train()
with open('raw_data_overfit.csv', 'w', newline= '') as csvfile:
    fieldname = ['overfit_train_acc', 'overfit_val_acc', 'overfit_loss', 'overfit_loss_val']
    writer = csv.DictWriter(csvfile, fieldnames= fieldname)

    for i in range(len(solver.train_acc_history)):
        writer.writerow({'overfit_train_acc' : solver.train_acc_history[i], 'overfit_val_acc' : solver.val_acc_history[i]})

    for i in range(len(solver.loss_history)):
        writer.writerow({'overfit_loss' : solver.loss_history[i], 'overfit_loss_val' : solver.loss_history_val[i]})



##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
