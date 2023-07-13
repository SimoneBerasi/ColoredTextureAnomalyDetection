from skimage import morphology
from skimage.color import lab2rgb

from ColorUtils import prepare_dataset_colorssim, prepare_image_colorssim
from DataLoader import load_patches_from_image
from NNModels import Model_noise_skip_bigger_old
import cv2
import numpy as np

from Prediction import image_reconstruction, suppress_borders_size
from Steerables.AnomalyMetrics import cw_ssimcolor_metric
from Utils import batch_evaluation, visualize_results

image_path = '/home/simo/Desktop/Thesis Projects/AnomalyDetectionBionda/Dataset/MVTec_Data/leather/test/cut/003.png'

threshold = 0.235


def suppress_anomaly_on_borders(anomaly_map):
    #b = anomaly_map.copy()
    mask_size = anomaly_map.shape[0]
    mask = np.ones((mask_size, mask_size), float)*1.3
    mask[suppress_borders_size:-suppress_borders_size, suppress_borders_size:-suppress_borders_size] = 1
    anomaly_map = np.divide(anomaly_map, mask)
    #visualize_results(b, anomaly_map, "before vs after border suppression")
    return anomaly_map

def circular_erosion(anomaly_map):
    #b = anomaly_map.copy()
    disk = morphology.disk(10)
    anomaly_map = morphology.opening(anomaly_map, disk)
    #visualize_results(b, anomaly_map, "before vs after erosion")
    return anomaly_map



if __name__ =='__main__':

    ae = Model_noise_skip_bigger_old((256, 256, 3))
    ae.summary()
    ae.load_weights('/home/simo/Desktop/Thesis Projects/AnomalyDetectionBionda/Weights/new_weights/mnsbiggercwssimc_simpler-leather-19600-140.h5')

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_size = 1024

    if image_size != 1024:
        pad_size = int((1024-image_size)/2)
        img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    patches, y_valid = load_patches_from_image(img, 256, random=False, stride = 16)
    print(patches.shape)
    y_valid_lab = prepare_image_colorssim(y_valid)
    patches = np.array(patches)
    patches1 = patches
    patches = prepare_dataset_colorssim(patches) / 102


    _, pred = batch_evaluation(patches, ae, 100)
    pred = np.array(pred)

    pred[:, :, :, 0] = pred[:, :, :, 0] * (-1)

    predicted_image_lab = image_reconstruction(pred, 64) * 102
    predicted_image = lab2rgb(predicted_image_lab)

    print(predicted_image)
    print(predicted_image_lab)

    resid_color, resid_cwssim, residual = cw_ssimcolor_metric(predicted_image_lab/102, y_valid_lab/102, 1024, 1024, False)




    if image_size == 840:
        resid_color = resid_color[92:1024-92, 92:1024-92] #used only for tile
        resid_color = np.reshape(resid_color, (840, 840))

        resid_cwssim = resid_cwssim[92:1024-92, 92:1024-92] #used only for tile
        resid_cwssim = np.reshape(resid_cwssim, (840, 840))

        residual = residual[92:1024-92, 92:1024-92] #used only for tile
        residual = np.reshape(residual, (840, 840))

        print(y_valid.shape)
        y_valid = y_valid[92:1024-92, 92:1024-92] #used only for tile
        print(y_valid.shape)
        y_valid = np.reshape(y_valid, (840, 840, 3))

        predicted_image = predicted_image[92:1024-92, 92:1024-92] #used only for tile
        predicted_image = np.reshape(predicted_image, (840, 840, 3))




    visualize_results(y_valid, predicted_image, "predicted" )

    #visualize_results(y_valid, resid_color, "color anomaly map" )
    #visualize_results(y_valid, resid_cwssim, "cwssim anbomaly map" )
    residual2 = circular_erosion(residual)
    visualize_results(residual, residual2, "anomaly map")
    #residual2 = suppress_anomaly_on_borders(residual2)
    #visualize_results(residual, residual2, "erosion")
    #visualize_results(residual2, residual2>threshold, "threshold")

#    for i in range(len(patches)):
 #       visualize_results(patches1[i], lab2rgb(pred[i]*102))

