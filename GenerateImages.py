
#This file is for generating images (like anomaly maps) to be used in the thesis

from NNModels import *
from DataLoader import *
from Evaluation import *
from Prediction import *


weights_file_path = ''
image_file_path = ''
ground_trth_path = ''




if __name__ == '__main__':

    autoencoder = Model_noise_skip_bigger_old()