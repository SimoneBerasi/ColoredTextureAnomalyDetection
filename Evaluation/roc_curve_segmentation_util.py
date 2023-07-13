from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
import numpy as np
import matplotlib.pyplot as plt

def compute_segmentation_roc(anomaly_maps, ground_truth_maps):
    


    anomaly_maps = np.array(anomaly_maps)
    ground_truth_labels = np.array(ground_truth_maps)/255

    #ground_truth_labels = ground_truth_labels.astype(int)

    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels, axis=0)

    #print(ground_truth_labels)
    #print(anomaly_maps)
    dims = anomaly_maps.shape

    anomaly_maps = np.reshape(anomaly_maps, dims[0]*dims[1] )
    ground_truth_labels = np.reshape(ground_truth_labels, dims[0] * dims[1])

    print(ground_truth_labels.shape)

    roc_score = roc_auc_score(ground_truth_labels,anomaly_maps, max_fpr=1)
    #roc_score = roc_auc_score(anomaly_maps,ground_truth_labels, max_fpr=0.3)
    #fpr, tpr, thresholds = roc_curve(ground_truth_labels, anomaly_maps)


    #display = RocCurveDisplay.from_predictions(ground_truth_labels, anomaly_maps)
    #display.plot()
    #plt.show()

    #fpr, tpr, ths = roc_curve(ground_truth_labels, anomaly_maps, pos_label=1)
    
    return roc_score#, #fpr, tpr
