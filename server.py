from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from urllib.request import Request, urlopen
from PIL import Image
from io import BytesIO
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
from test import main
import base64
import cv2

app = Flask(__name__)

CORS(app)
#API Routes

@app.route("/")
def hello():
    return "Super resolution using GAN"

@app.route('/derain', methods=['POST'])
def superResolution():
    try:
        data = request.get_json()
        image_url = str(data.get('url'))
        print(image_url)
        req = Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urlopen(req)
        lowQualityImg = response.read()
        with open('LR/rain_image.png', 'wb') as f:
            f.write(lowQualityImg)
        fname = 'lowQualityImg'
        main('./LR/rain_image.png')
        result = {'url': f'{image_url}'}


        img1 = mpimg.imread(f'./results/rlt.png')
        img2 = mpimg.imread(f'./LR/rain_image.png')

        # Resize img2 to match the height of img1
        img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        combined_img = np.hstack((img2_resized, img1))
        plt.imsave(f'./results/{fname}_combined.png', combined_img)
        
        # Open the image file in binary mode
        with open(f'./results/{fname}_combined.png', 'rb') as f:
            # Read the file and encode it as base64
            image_data = base64.b64encode(f.read())
            # Decode the base64 bytes to ASCII
            base64_string = image_data.decode('ascii')
        
        result = {'image': base64_string}
        
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify(e), 400

if __name__ == "__main__":
    app.run(port=5000,debug=True)