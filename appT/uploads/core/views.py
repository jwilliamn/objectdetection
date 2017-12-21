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


def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


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
        succ, pred = detection(image, prototxt, model)
        print("status: ", succ)
        if succ == 1:
            return render(request, 'core/process.html', {
                'uploaded_file_url': uploaded_file_url,
                'pred': pred
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
    return succ, os.path.join(settings.OUT_URL, "predicted.png")
