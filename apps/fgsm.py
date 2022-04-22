import streamlit as st
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import PIL.Image as Image

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications import MobileNetV2

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
def app():
    class GradCAM:
        def __init__(self, model, classIdx, layerName=None):
            self.model = model
            self.classIdx = classIdx
            self.layerName = layerName
            if self.layerName is None:
                self.layerName = self.find_target_layer()

        def find_target_layer(self):
            for layer in reversed(self.model.layers):
                if len(layer.output.shape) == 4:
                    return layer.name
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

        def compute_heatmap(self, image, eps=1e-8):
            gradModel = Model(inputs=[self.model.inputs], outputs= [self.model.get_layer(self.layerName).output, self.model.output])
            with tf.GradientTape() as tape:
                inputs = tf.cast(image, tf.float32)
                (convOutputs, predictions) = gradModel(inputs)
                loss = predictions[:, self.classIdx]
            grads = tape.gradient(loss, convOutputs)
            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads
            convOutputs = convOutputs[0]
            guidedGrads = guidedGrads[0]
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))
            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            heatmap = (heatmap * 255).astype("uint8")
            return heatmap

        def overlay_heatmap(self, heatmap, image, alpha=0.5,
            colormap=cv2.COLORMAP_JET):
            heatmap = cv2.applyColorMap(heatmap, colormap)
            output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
            return (heatmap, output)

    loss_object = tf.keras.losses.CategoricalCrossentropy()

    def create_adversarial_pattern(input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
        return signed_grad

    def preprocess(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = image[None, ...]
        return image

    # Helper function to extract labels from probability vector


    def display_images(image, description):
        _, label, confidence = get_imagenet_label(model.predict(image))
        plt.figure()
        plt.imshow(image[0]*0.5+0.5)
        plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
        plt.show()

    st.set_option('deprecation.showfileUploaderEncoding', False)

    @st.cache(allow_output_mutation=True)
    def load_model():
        pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
        pretrained_model.trainable = False
        return pretrained_model

    model=load_model()
    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

    st.write(""" #### Fast Gradient Sign Method """)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])

    def get_imagenet_label(probs):
        return decode_predictions(probs, top=1)[0][0]

    def importAnalyze(image_data, model, eps, alpopacity):
        resized = cv2.resize(image_data, (224, 224))
        image = preprocess(image_data)
        image_probs = model.predict(image)

        predictions = model.predict(image)
        pred_i = np.argmax(predictions[0])
        (_, labelClass, confidence) = get_imagenet_label(model.predict(image))

        cam = GradCAM(model, pred_i)
        heatmap = cam.compute_heatmap(image)
        heatmap = cv2.resize(heatmap, (image_data.shape[1], image_data.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, image_data, alpha=alpopacity)

        image_class_index = pred_i
        label = tf.one_hot(image_class_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        perturbations = create_adversarial_pattern(image, label)

        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)
        heatmap_adv = cam.compute_heatmap(adv_x)
        heatmap_adv = cv2.resize(heatmap_adv, (image_data.shape[1], image_data.shape[0]))
        (heatmap_adv, output_adv) = cam.overlay_heatmap(heatmap_adv, image_data, alpha=alpopacity)
        (_, labeladv, confidenceadv) = get_imagenet_label(model.predict(adv_x))
        attacked = np.squeeze(adv_x*0.5+0.5)
        attacked = cv2.resize(attacked, (image_data.shape[1], image_data.shape[0]))
        ptr = np.squeeze(perturbations*0.5+0.5)
        ptr = cv2.resize(ptr, (image_data.shape[1], image_data.shape[0]))
        return (heatmap, output, labelClass, confidence, heatmap_adv, output_adv, labeladv, confidenceadv, attacked, ptr)

    if file is None:
        pass
    else:
        image = Image.open(file)
        ops = st.slider("Overlay opacity:", min_value=0.0, max_value=1.0, step=0.01, value=0.7)
        eps = st.slider("Choose epsilon value: ", min_value=0.0, max_value=5.0, step=0.01, value=0.05)
        
        (heat, overlay, labelorig, confidence, heatadv, overadv, labeladv, confidenceadv, attacked, perturbation) = importAnalyze(np.array(image), model, eps, ops)
        conf = "{:.2f}".format(confidence * 100)
        confadv = "{:.2f}".format(confidenceadv * 100)

        my_expander = st.expander(label='View attack composition')
        my_expander.write('fgsm attack composition')
        with my_expander:
            col1, col2, col3 = st.columns(3)
            col1.header("Original")
            col2.header("Perturbation")
            col3.header("Attacked")

            col1.image(image)
            col2.image(perturbation)
            col3.image(attacked)

        col4, col5 = st.columns(2)
        col4.header("Original image")
        col4.image(image, use_column_width=True)

        col5.header("Attacked image")
        col5.image(attacked, use_column_width=True)

        col4.image(heat)
        col4.image(overlay, caption=labelorig+" "+str(conf)+"%")
        col5.image(heatadv)
        col5.image(overadv, caption=labeladv+" "+str(confadv)+"%")
