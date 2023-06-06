import cv2
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, Response
from src.ASLModel import ASLModel
from src.ASLPredictor import ASLPredictor
from src.ASLDetector import ASLDetector

app = Flask(__name__)
video = cv2.VideoCapture(0)

def gen(video):
    model = ASLModel(24, "cpu")
    predictor = ASLPredictor(model, "./assets/model_3.pt", "cpu")
    detector = ASLDetector(predictor=predictor)
    while True:
        success, image = video.read()
        if not success:
            print("Unable to read frame")
            break
        
        # convert the image to correct format for mediapipe
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # detect the hand and display the result
        annotated_frame = detector.detect(image)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        success, annotated_frame = cv2.imencode('.jpg', annotated_frame)
        if not success:
            print("Unable to encode frame")
            break
        
        # yield the frame in byte format
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame.tobytes() + b'\r\n\r\n')
        
@app.route('/')
def index():
    global video
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2207, threaded=False)