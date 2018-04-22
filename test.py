import numpy as np
import data_utils as dl
import eval_nn as TEST_NN
from utils import solver





def test_fer_model(img_folder, model = ''):

    #start = 28710
    #end = 32298

    solver = dl.loadModelNN(model)


    #test_data = dl.loadDatafromFolderINX(img_folder, start, end)

    test_data = dl.loadDataSorted(img_folder)

    miu = dl.loadDatafrPKL('miu.pkl')  # miu is for data regularization, you could generate your own one
    test_data -= miu

    predictions = TEST_NN.eval_nn(solver, test_data)

    return predictions


if __name__ == '__main__':

    img_folder = ''

    predictions = test_fer_model(img_folder)

    #DATADIC = dl.loadDatafrPKL('DATA_CSV.pkl')

    #y_eval = DATADIC['y_test']

    #print (np.mean(y_eval == predictions))
