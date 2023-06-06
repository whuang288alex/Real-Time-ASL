import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.ASLModel import ASLModel

class ASLPredictor():
    
    class Timer():
        def __init__(self, duration):
            self.tik = time.time()
            self.tok = time.time()
            self.duration = duration
            
        def times_up(self):
            self.tok = time.time()
            if self.tok - self.tik > self.duration:
                self.tik = self.tok
                return True
            return False
    
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    def __init__(self, model, checkpoint_path, device, duration=1):
        self.model = model
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.eval()
        self.predicted = None
        self.timer = self.Timer(duration)
    
    def predict(self, img):
        if self.timer.times_up():
            # transform the image to the correct format
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
            img = transforms.Resize((28, 28))(img)
            img = img.unsqueeze(0)
            img /= 255.0
            
            # predict the letter
            output = self.model(img)
            predicted = torch.softmax(output,dim=1) 
            _, predicted = torch.max(predicted, 1) 
            predicted = predicted.cpu() 
            idx = predicted[0]
            self.predicted = self.letters[idx]
            
        return self.predicted
