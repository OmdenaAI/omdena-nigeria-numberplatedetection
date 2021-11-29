#use python3 main.py --image_dir "PATH TO FILE"
#save the image to this folder you want to use
import cv2
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
from paddleocr import PaddleOCR
import argparse, logging, os

wpod_net_path = "wpod-net.json"
img_path = 'np.png'
parser = argparse.ArgumentParser()
logging.basicConfig(filename="anpr.log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser.add_argument('-i', '--image_dir', type=str, required=True, help='path to image')
args = parser.parse_args()

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        logger.info("Loaded model successfully...")
        return model
    except Exception as e:
        print(e)

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
    # for line in result:
    #     logger.info(line)
    return result

def clean_text(recognized):
    upperRec = recognized.upper()
    ans = ''
    for x in upperRec:
        if (ord(x)>=60 and ord(x)<=90) or (ord(x)>=48 and ord(x)<=57):
            ans += x
    return ans

if __name__ == "__main__":
    wpod_net = load_model(wpod_net_path)

    #update the image path to run
    image_path = args.image_dir

    LpImg,cor = get_plate(image_path)
    print("Detected %i plate(s) in"%len(LpImg),splitext(basename(image_path))[0])
    print("Coordinate of plate(s) in image: \n", cor)

    #write image to directory
    img = LpImg[0]
    cv2.imwrite('np.png', 255*img)
    img = preprocess_image(img_path, resize=True, threshold=False)
    bfilter = cv2.bilateralFilter(img, 11, 17, 17)
    cv2.imwrite('np.png', bfilter)
    result = recognizePlate(img_path)
    ans=[]
    for line in result:
        if len(line[1][0]) >= 7:
            ans.append(clean_text(line[1][0]))
    logger.info(ans)