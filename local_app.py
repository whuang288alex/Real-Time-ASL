import cv2
import warnings
warnings.filterwarnings("ignore")
from src.ASLModel import ASLModel
from src.ASLPredictor import ASLPredictor
from src.ASLDetector import ASLDetector

vid = cv2.VideoCapture(0)

if __name__ == "__main__":
    model = ASLModel(24, "cpu")
    predictor = ASLPredictor(model, "./assets/model_3.pt", "cpu")
    detector = ASLDetector(predictor=predictor)
    while(True):
        # Capture the video frame and convert to mediapipe image
        success, image = vid.read()
        if not success:
            print("Unable to read frame")
            break
        
        # convert the image to correct format for mediapipe
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # detect the hand and display the result
        annotated_frame = detector.detect(image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Real-Time-ASL', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()