import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the TensorFlow model
Model_Enhancer = load_model("path_to_your_model")

# Preprocessing functions
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.0001
        sigma = var ** 0.05
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = gauss + image
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 1.0
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

def ExtractTestInput(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_ = cv.resize(img, (500, 500))
    hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * 0.2
    img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    noise_img = noisy("s&p", img1)
    noise_img = noise_img.reshape(1, 500, 500, 3)
    return noise_img

# Streamlit app
def main():
    st.title("Image Enhancement App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Preprocess and make prediction
        if st.button("Enhance Image"):
            # Preprocess the uploaded image
            img = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_ = cv.resize(img, (500, 500))
            hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV)
            hsv[..., 2] = hsv[..., 2] * 0.2
            img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            noise_img = noisy("s&p", img1)
            noise_img = noise_img.reshape(1, 500, 500, 3)

            # Make prediction using the model
            prediction = Model_Enhancer.predict(noise_img)

            st.image(prediction[0], caption="Enhanced Image", use_column_width=True)

if __name__ == "__main__":
    main()
