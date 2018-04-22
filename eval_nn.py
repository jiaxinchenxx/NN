import numpy as np
from utils import solver
from utils import optim

def eval_nn(solver, data):

    scores = solver.model.loss(data)
    y_pred = np.argmax(scores, axis = 1)
    return y_pred
