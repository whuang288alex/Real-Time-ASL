import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
import torch.nn as nn

class ASLModel(nn.Module):
    def __init__(self, num_class, device):
        super(ASLModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_class)
        self.model = self.model.to(device)

    def forward(self, x):
        x = self.model(x)
        return x
    
model = ASLModel(26, torch.device('cpu'))
model.load_state_dict(torch.load("./ckpt/model_1.pt",  map_location=torch.device('cpu')))
model.eval()
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def predict(img):
    # transform the input image
    img = T.Grayscale()(img)
    img = T.Resize((224, 224))(img)
    img = img.numpy()
    img = torch.FloatTensor(img)
    img = torch.unsqueeze(img, 0)
    img /= 255.
    
    # predict based on the net
    output = model(img)
    predicted = torch.softmax(output,dim=1) 
    _, predicted = torch.max(predicted, 1) 
    predicted = predicted.data.cpu() 
    idx = predicted[0]
    
    # return the predicted character
    return letters[idx]

def translate():
    hands = mp.solutions.hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    while True:

        # process keyboard input
        k = cv2.waitKey(1)
        capture = False
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        if k % 256 == 32:
            capture = True
        
        # get the frame and locate hands
        _, frame = cap.read()
        h, w, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        
        # if there is a hand detected
        if hand_landmarks:
            index = 0
            # for each hand detected (maximum 2)
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                # For each hand, there will be 21 points
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                
                # make sure the hand frame doesn't go out of bound    
                x_max = max(w, x_max)
                y_max = max(h, y_max)
                x_min = min(0, x_min)
                y_min = min(0, y_min)
                if y_max - y_min <= 0 or x_max - x_min <= 0:
                    continue
                
                # draw a rectangle around the detected hands and show the captured hands landmarks
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(frame, handLMs, mp.solutions.hands.HAND_CONNECTIONS)

                # make prediction based on hand gesture and show the result
                hand_frame = framergb[y_min:y_max, x_min:x_max]
                hand_frame_tensor = torch.permute(torch.from_numpy(hand_frame), (2,0,1))
                frame = cv2.putText(frame,  predict(hand_frame_tensor), (100,100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                
                # if capture:
                #     cv2.imwrite(os.path.join("captured", "hand{}.jpg".format(index)), hand_frame)
                #     index += 1    
        cv2.imshow("Frame", frame)    
    cap.release()
    
if __name__ ==  '__main__':
    translate()