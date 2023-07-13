
from NNModels import Model_noise_skip, Model_noise_skip_big, Model_noise_skip_bigger, Model_noise_skip_color_only, Model_noise_skip_bigger_old
from ColorUtils import *
from Utils import *
from PIL import Image
from sklearn import metrics
from skimage.color import lab2rgb, rgb2lab, gray2rgb, rgb2gray
from ColorUtils import *
from Steerables.AnomalyMetrics import cw_ssim_metric, ssim_metric, l2_metric, cw_ssimcolor_metric
from skimage import morphology

img_size = 1024
ae_patch_size = 256
ae_stride = 16
ae_batch_splits = 50
anomaly_metrics = 'color_cwssim_loss'
suppress_borders = True
apply_circular_erosion = True
suppress_borders_size = 30


test_dir = "/home/simo/Desktop/Thesis Projects/AnomalyDetectionBionda/Dataset/MVTec_Data/carpet/test"
anomaly_maps_directory = "out/Maps"
weights_file = 'Weights/new_weights/ccwssim_carpet_128_fixed200.h5'


if __name__ == '__main__':


    vailed_ext = [".jpg", ".png"]
    import os

    f_list = []

    def Test2(rootDir):
        for lists in os.listdir(rootDir):
            path = os.path.join(rootDir, lists)
            filename, file_extension = os.path.splitext(path)
            if file_extension in vailed_ext:
                print(path)
                f_list.append(path)
            if os.path.isdir(path):
                Test2(path)

    Test2(test_dir)

    imgs = []
    for path in f_list:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    imgs = np.array(imgs)
    print(imgs.shape)

    imgs = prepare_dataset_colorssim(imgs)/100






    #autoencoder = Model_noise_skip_bigger_old(input_shape=(None, None, 3))
    autoencoder = Model_noise_skip_bigger_old(input_shape=(None, None, 3))
    #autoencoder = Model_noise_skip_bigger(input_shape=(None, None, 3))
    autoencoder.load_weights(weights_file)

    print("prima")
    pred = autoencoder(imgs[0:50])
    print("dopo")
