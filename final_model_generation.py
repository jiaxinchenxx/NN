import numpy as np
import matplotlib.pyplot as plt
from fcnet import FullyConnectedNet
from utils.solver import Solver
from PIL import Image
from sklearn.model_selection import train_test_split



# from fcnet import FullyConnectedNet
# from utils.solver import Solver
# from utils.data_utils import get_CIFAR10_data
import data_utils as dl
# from PIL import Image


DATADIC = dl.loadDatafrPKL('DATA_NEW.pkl')
#TESTDIC = dl.loadDatafrPKL('DATA_CSV.pkl')

train_data=DATADIC['x_train']
train_labels=DATADIC['y_train']

eval_data = DATADIC['x_test']
eval_labels = DATADIC['y_test']

x_train = np.append(train_data, eval_data, axis = 0)
y_train = np.append(train_labels, eval_labels)

DATADICT = dl.loadDatafrPKL('DATA_CSV.pkl')

eval_data = DATADICT['x_test']
eval_labels = DATADICT['y_test']

data = {
    'X_train' : x_train,
    'y_train' : y_train,
    'X_val' : eval_data,
    'y_val' : eval_labels
}

miu = np.mean(data['X_train'], axis = 0)
dl.saveData(miu, 'miu.pkl')
data['X_train'] -= miu
data['X_val'] -= miu

model = FullyConnectedNet([1024, 512], num_classes = 7, input_dim=48*48, dropout = 0.3, reg = 0.01,
                weight_scale = 1e-2, dtype = np.float32, seed = 42)

solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config = {
                    'learning_rate' : 0.0005,
                    'momentum' : 0.92},
                    #'momentum': 0.95},
                #optim_config={
                #    'learning_rate': 0.0012,
                #    'beta1': 0.9,
                #    'beta2': 0.995,
                #    'epsilon': 1e-8,
                #    't': 0 },
                #optim_config={
                #'learning_rate': 0.00222,
                #'momentum': 0.21
                #},
                lr_decay = 1.0,
                verbose=True,
                num_epochs=75, batch_size=128,
                print_every=100, checkpoint_name = 'final_model_nn_SUBMIT')

solver.train()



# scores=model.loss(eval_data)
# prediction_label=np.argmax(scores,axis=1)
# confusion_matrix=Confusion_matrix(prediction_label, eval_labels, 7)
# print(confusion_matrix)
# dl.plot_cm(confusion_matrix)
#     #recall,precision=normalised_recall_precision(confusion_matrix)
#     #f1=F1_measure(recall,precision)
#     #print(f1)
#
# plt.subplot(2, 1, 1)
# plt.title('Training loss')
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('Iteration')
#
#         # loss, grads =solver.model.loss(X_batch, y_batch)
#         # solver.loss_history.append(loss)
#
# plt.subplot(2, 1, 2)
# plt.title('Accuracy')
# plt.plot(solver.train_acc_history, '-o', label='train')
# plt.plot(solver.val_acc_history, '-o', label='val')
# plt.plot([0.5] * len(solver.val_acc_history), 'k--')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.gcf().set_size_inches(15, 12)
# plt.show()
