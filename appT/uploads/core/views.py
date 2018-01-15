from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage, default_storage
from django.core.files.base import ContentFile


from uploads.core.models import Document
from uploads.core.forms import DocumentForm

import numpy as np
import argparse
import cv2
import os

from os import listdir
from os.path import isfile, join

from uploads.core.FastRCNN_train import prepare, train_fast_rcnn
from uploads.core.FastRCNN_eval import compute_test_set_aps, FastRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

# Initial settings
#image = "images/test.jpg"
prototxt_ = "MobileNetSSD_deploy.prototxt.txt"
model_ = "MobileNetSSD_deploy.caffemodel"
output = settings.OUT_ROOT

# Class labels taken from PASCAL VOC dataset
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#CNTK settings
path_out = "uploads/core/Output/Grocery/"
out_dir = os.path.join(settings.BASE_DIR, path_out)

def get_configuration():
    # load configs for detector, base network and data set
    from uploads.core.FastRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# trains and evaluates a Fast R-CNN model.
#if __name__ == '__main__':
def cntkDemo():
    cfg = get_configuration()
    prepare(cfg, True)

    # train and test
    trained_model = train_fast_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FastRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

# Views
def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def demo_cntk(request):
    #cntkDemo()
    if out_dir:
        print("out_dir",  out_dir)
        img_list = [f for f in listdir(out_dir) if isfile(join(out_dir, f))]
        print("Images:", img_list)        
        
        return render(request, 'core/results.html', {
            'out_dir': out_dir,
            'images': img_list
            })
        
    return render(request, 'core/error_cntk.html')

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        
        image = os.path.join(settings.MEDIA_ROOT, str(filename))
        prototxt = os.path.join(settings.RES_ROOT, prototxt_)
        model = os.path.join(settings.RES_ROOT, model_)
        print("settings.RES_ROOT",  settings.RES_ROOT)
        print("settings.RES_ROOT", model)
        succ, pred, inix, iniy, size = detection(image, prototxt, model)
        print("status: ", succ)
        if succ == 1:
            return render(request, 'core/process.html', {
                'uploaded_file_url': uploaded_file_url,
                'pred': pred,
                'inix':inix,
                'iniy': iniy,
                'size':size
            })
        else:
            return render(request, 'core/error.html', {
                'uploaded_file_url': uploaded_file_url
            })
    return render(request, 'core/simple_upload.html')

def process(request):
    if request.method == 'GET':
        #fs = FileSystemStorage()
        #filename = fs.open()
        return render(request, 'core/process.html', {
            'uploaded_file_url': uploaded_file_url
        })

    return render(request, 'core/home.html')

def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })

def detection(image, prototxt, model):
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Loading image and resize to feed the network
    image = cv2.imread(image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # Detection and prediction
    net.setInput(blob)
    detections = net.forward()

    succ = 0
    ini_x, ini_y = 0, 0
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # get index of label
            id_label = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7]*np.array([w, h, w, h])
            (ini_x, ini_y, end_x, end_y) = box.astype("int")

            print("poss: ", ini_x, ini_y, end_x, end_y)
            label = CLASSES[id_label] + ": " + str(confidence*100)

            #display
            cv2.rectangle(image, (ini_x, ini_y), (end_x, end_y), COLORS[id_label], 2)
            if ini_y - 10 > 10:
                y = ini_y - 10
            else:
                y = ini_y + 10
            newI = cv2.putText(image, label, (ini_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[id_label], 2)
            succ = 1
        else:
            succ = 0

    cv2.imwrite(os.path.join(output, "predicted.png"), image)
    #plt.imshow(newI, cmap=plt.cm.gray)
    #plt.show()
    return succ, os.path.join(settings.OUT_URL, "predicted.png"), ini_x, ini_y, (end_y-ini_y)
