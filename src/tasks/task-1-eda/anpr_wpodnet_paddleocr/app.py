from flask import Flask, render_template, request, redirect, url_for, jsonify
import re, base64
import numpy as np
import cv2
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
from paddleocr import PaddleOCR

app = Flask(__name__)

wpod_net_path = "wpod-net.json"
image_path = 'np.png'
img_path = 'np.png'

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loaded model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False, threshold=True):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if threshold:
        img = img / 255
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if resize:
        img = cv2.resize(img, (180,80))
    return img

def get_plate(image_path, Dmax=608, Dmin=156):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = max(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

def recognizePlate(image_path):
    ocr = PaddleOCR(show_log=False)
    result = ocr.ocr(image_path)
    return result

def clean_text(recognized):
    upperRec = recognized.upper()
    ans = ''
    for x in upperRec:
        if (ord(x)>=60 and ord(x)<=90) or (ord(x)>=48 and ord(x)<=57):
            ans += x
    return ans


def predict_fit(image_data):
    img = base64.b64decode(image_data)
    arr = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(arr, flags=cv2.IMREAD_COLOR)
    cv2.imwrite('np.png', img)

    try:
        LpImg,cor = get_plate(image_path)
    except Exception as e:
        print("Dmax value is not fit changing to 208")
        LpImg, cor = get_plate(image_path, Dmax=208)

    img = LpImg[0]
    cv2.imwrite('np.png', 255*img)
    img = preprocess_image(img_path, resize=True, threshold=False)
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)
    cv2.imwrite('np.png', bfilter)
    result = recognizePlate(img_path)
    ans=[]
    for line in result:
        if len(line[1][0]) >= 7:
            text = clean_text(line[1][0])
            if text.isalnum() and not text.isalpha():
                ans.append(text)
    if ans==[]:
        return "No Number Plate Found"
    return ans[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_base64 = request.values['imageBase64']
        #print("predict" + img_base64)
        image_data = re.sub('^data:image/.+;base64,', '', img_base64)

        text = predict_fit(image_data)
        results = {'digit': str(text)}
        return jsonify(results)
    else:
        return "Error occured"


if __name__ == '__main__':
    app.run(debug=True)
