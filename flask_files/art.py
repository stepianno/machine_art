import flask
from flask import render_template, request, Flask
from keras.models import Input
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from scipy import spatial
import numpy as np
import os
import cv2
import pickle
from skimage import io

app = Flask(__name__)
app.debug=True

art = os.listdir('static/moma')
art.remove('.DS_Store')

features1 = pickle.load(open('art_features1.pickle', 'rb'))
features2 = pickle.load(open('art_features2.pickle', 'rb'))
features = np.concatenate((features1, features2), axis=0)
feature_extractor = load_model('feature.h5')


@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/encode')
def autoencoder():
    return render_template('autoencoder.html')

@app.route('/image')
def imager():
    return render_template('imager.html')

@app.route('/image2')
def image2():
    return render_template('imager2.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')

@app.route('/recommend', methods=['post','get'])
def recommender():
    image = request.form.get('link')
    if not image:
        i = np.random.choice(len(art))
        image = f'static/moma/Vincent-van-Gogh_The-Starry-Night_Saint-Rémy,-June-1889.jpeg'
        image_array = cv2.imread(image)
    else:
        image_array = io.imread(image)
    try:
        x = cv2.resize(image_array, (224, 224))
        x = x / 255
        y = feature_extractor.predict(x.reshape(1, 224, 224, 3))
        y = y.reshape(1, np.prod(y.shape))
        dist = spatial.distance.cdist(y, features, metric='cosine')[0]
        if min(dist)<0.00001:
            ind = dist.argsort()[1:4]
        else:
            ind = dist.argsort()[:3]
        works = [art[ind[0]], art[ind[1]], art[ind[2]]]
    except:
        works = []
    return render_template('recommender.html', image=image, works=works)

@app.route('/fin')
def end():
    return render_template('end.html')


if __name__ == '__main__':
    app.run()