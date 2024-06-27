from flask import Flask, render_template, request
import pandas as pd
from keras import Sequential
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from sklearn.metrics.pairwise import cosine_similarity
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")
# Load data and model
df = pd.read_csv('styles_image_paths.csv', nrows=5000)
df['text'] = df['gender'] + ' ' + df['masterCategory'] + ' ' + df['subCategory'] + ' ' + df['articleType'] + ' ' + df[
    'baseColour'] + ' ' + df['season'] + ' ' + df['usage'] + ' ' + df['productDisplayName']
df.fillna("", inplace=True)

model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(df['text'].tolist())


# Function to load and resize image
def load_image(img_path):
    with Image.open(img_path) as img:
        img = img.resize((80, 60))  # Resize to 80x60 pixels
        return img


# Function to generate image plot
def plot_image(img_data):
    img_buf = BytesIO()
    img_data.save(img_buf, format='PNG')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    return img_str


# Compute pairwise cosine similarities
cosine_similarities = cosine_similarity(embeddings)
n = 3

# Find top n similar products for each product
similar_products = {}
for i, row in enumerate(cosine_similarities):
    similar_indices = np.argsort(-row)[1:n + 1]
    similar_ids = df.iloc[similar_indices].index.tolist()
    similar_products[df.index[i]] = similar_ids

app = Flask(__name__)
# Load the pre-trained GAN generator model
generator_model_path = 'gan_generator.h5'

def discriminator(rows, cols, channels):
    shape = (rows, cols, channels)
    model = Sequential()
    model.add(Flatten(input_shape = shape))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha = 0.01))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    if request.method == 'POST':
        user_text = request.form['description']
        user_embedding = model.encode([user_text])[0]
        similarities = cosine_similarity([user_embedding], embeddings)[0]
        similar_indices = np.argsort(-similarities)[:n]

        for index in similar_indices:
            img_data = load_image(df.iloc[index]['image'])
            img_str = plot_image(img_data)
            images.append(img_str)
    return render_template('index.html', images=images)


if __name__ == '__main__':
    app.run(debug=True)
