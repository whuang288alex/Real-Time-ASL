import os
import cv2
import torch
import mediapipe as mp
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# define the model
class ASLModel(nn.Module):
    def __init__(self, num_class, device):
        super(ASLModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_class)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x
    
class Predictor():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    def __init__(self, checkpoint_path, device):
        self.model = ASLModel(26, device, checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
    
    def predict(self, img):
        # predict based on the net
        output = self.model(img)
        predicted = torch.softmax(output,dim=1) 
        _, predicted = torch.max(predicted, 1) 
        predicted = predicted.data.cpu() 
        idx = predicted[0]
        # return the predicted character
        return self.letters[idx]

# show the webcam
vid = cv2.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()