from __future__ import print_function, division
import cv2
import sys
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import cv2
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class Net(nn.Module):
  def __init__(self):
        super(Net, self).__init__()
        # Transforms 3 x 128 x 128 to 8 x 124 x 124
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)

        # Transforms 32 x 124 x 124 to 64 x 120 x 120
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        # Transforms 64 x 60 x 60 to 128 x 60 x 60
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)

        self.fc1 = nn.Linear(100352, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 2)

  def forward(self, x):
        # CONVNET
        x = F.relu(self.conv1(x))
        x = F.relu((self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu((self.conv3(x)))
        
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # FC
        x = x.view(x.size(0), -1) # Flatten the image for Fully Connected Layer
        #print(x.shape)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x))) 
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)

#from google.colab import drive

font = cv2.FONT_HERSHEY_SIMPLEX

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

model = torch.load('modelFaceMask.pt')
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = ['Mask', 'No Mask']
labelColor = [ (10, 255, 0),(10, 0, 255)]
transformations = Compose([
    ToPILImage(),
    Resize((128, 128)),
    ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])



while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for face in faces:
        xStart, yStart, width, height = face
        
        # clamp coordinates that are outside of the image
        xStart, yStart = max(xStart, 0), max(yStart, 0)
        
        # predict mask label on extracted face
        faceImg = frame[yStart:yStart+height, xStart:xStart+width]
        output = model(transformations(faceImg).unsqueeze(0).to(device))
        _, predicted = torch.max(output.data, 1)

        # print(predicted)

        # draw face frame
        cv2.rectangle(frame,
                      (xStart, yStart),
                      (xStart + width, yStart + height),
                      (126, 65, 64),
                      thickness=2)

        # center text according to the face frame
        textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
        textX = xStart + width // 2 - textSize[0] // 2

        # draw prediction label
        cv2.putText(frame,
                    labels[predicted],
                    (textX, yStart-20),
                    font, 1, labelColor[predicted], 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
