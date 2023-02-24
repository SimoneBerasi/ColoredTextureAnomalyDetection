
from NNModels import Model_noise_skip, Model_noise_skip_big, Model_noise_skip_bigger
from ColorUtils import *
from Utils import *
from PIL import Image
from sklearn import metrics
from skimage.color import lab2rgb, rgb2lab
from ColorUtils import *
from Steerables.AnomalyMetrics import cw_ssim_metric, ssim_metric, l2_metric, cw_ssimcolor_metric


'''
Compute the anomaly maps and save them in a directory. #TODO define it

'''

img_size = 1024
ae_patch_size = 256
ae_stride = 16
ae_batch_splits = 100
anomaly_metrics = 'color_cwssim_loss'


test_dir = "Dataset/MVTec_Data/leather/test"
anomaly_maps_directory = "out/Maps"
weights_file = 'WeightsFile = Weights/new_weights/check_epoch140.h5'


def image_reconstruction(y_valid):

    reconstrunction = np.zeros((img_size, img_size, 3))  # The 3 are added only for color images
    normalizator = np.zeros((img_size, img_size, 3))

    i = 0
    j = 0
    for idx in range(len(y_valid)):
        reconstrunction[j:j + ae_patch_size, i:i + ae_patch_size] += y_valid[idx]
        normalizator[j:j + ae_patch_size, i:i + ae_patch_size] += np.ones((ae_patch_size, ae_patch_size, 3))

        if (i + ae_patch_size < img_size):
            i = i + ae_stride
        else:
            i = 0;
            j = j + ae_stride
    reconstrunction = reconstrunction / normalizator

    return reconstrunction


def compute_anomaly_map(image_path, autoencoder, patch_size, image_size, stride, max, invert_first_axis = False, pad_size = 0):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    patches, y_valid = load_patches_from_image(img, patch_size, random=False, stride = stride)
    patches = np.array(patches)
    patches = prepare_dataset_colorssim(patches) / max


    _, pred = batch_evaluation(patches, autoencoder, ae_batch_splits)
    pred = np.array(pred)

    if invert_first_axis:
        pred[:, :, :, 0] = pred[:, :, :, 0] * (-1)


    #todo restore
    predicted_image_lab = image_reconstruction(pred)*max
    #predicted_image_lab = pred*max

    print(" min max prediction")
    print(np.min(predicted_image_lab/max))
    print(np.max(predicted_image_lab/max))
    y_valid_lab = rgb2lab(y_valid)
    predicted_image_lab[:,:,0] = predicted_image_lab[:,:,0] * (-1)
    #predicted_image_lab[:,:,2] = predicted_image_lab[:,:,2] * (-1)

    predicted_image = lab2rgb(predicted_image_lab)
    print(predicted_image_lab)
    #visualize_results(predicted_image/255, predicted_image/255, 'a')

    valid_img = prepare_image_colorssim(y_valid)/max
    experimental_residual_total, residual_cwssim, residual_total = cw_ssimcolor_metric(predicted_image_lab/max, valid_img, image_size, image_size, False, pad_size = 0)

    #return residual_total
    return residual_total

def predict():
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


    autoencoder = Model_noise_skip_bigger(input_shape=(None, None, 3))
    autoencoder.load_weights(weights_file)

    i = 0
    total_max_color_loss = 0
    for item in f_list:
        anomaly_map = compute_anomaly_map(item, autoencoder, 256, 1024, ae_stride, 101)
        if np.max(anomaly_map) > total_max_color_loss:
            total_max_color_loss = np.max(anomaly_map)
        anomaly_map = np.array(anomaly_map)

        # anomaly_map = anomaly_map / np.max(anomaly_map)
        anomaly_map = np.reshape(anomaly_map, (1024, 1024))
        # anomaly_map = anomaly_map - np.min(anomaly_map)
        img = Image.fromarray(anomaly_map)
        path = os.path.splitext(item)[0]

        img.save(anomaly_maps_directory + "/" + path[path.find("MVTec")::] + ".tiff")

        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        visualize_results(img, anomaly_map, "a")
        i = i + 1
        print(i)

    print(anomaly_map)
    print("total color loss = ")
    print(total_max_color_loss / len(f_list))

