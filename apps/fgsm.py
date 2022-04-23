import streamlit as st
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import cv2
from cv2 import cvtColor, COLOR_RGB2BGR

class AdversarialImageGenerator:
    @staticmethod
    def create_adversarial_pattern(input_image, input_label, model):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad

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
        gradModel = tf.keras.models.Model(inputs=[self.model.inputs], outputs= [self.model.get_layer(self.layerName).output, self.model.output])
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

class Preprocessor:
    @staticmethod
    def preprocess(image, model):

        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        
        if(model == "ResNet50"):
            image = tf.keras.applications.resnet50.preprocess_input(image)
            
        elif(model == "MobileNetV2"):
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        elif(model == "VGG16"):
            image = tf.keras.applications.vgg16.preprocess_input(image)

        image = image[None, ...]
        return image

class Analyzer:
    @staticmethod
    def processAttack(image, image_data, model, eps, opacity, modelName, decode_predictions):
        image = Preprocessor.preprocess(image_data, modelName)
        predictions = model.predict(image)
        pred_index = np.argmax(predictions[0])
        (_, labelClass, confidence) = decode_predictions(predictions, top=1)[0][0]
        cam = GradCAM(model, pred_index)
        heatmap = cam.compute_heatmap(image)
        heatmap = cv2.resize(heatmap, (image_data.shape[1], image_data.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, image_data, alpha=opacity)
        label = tf.one_hot(pred_index, predictions.shape[-1])
        label = tf.reshape(label, (1, predictions.shape[-1]))
        
        perturbations = AdversarialImageGenerator.create_adversarial_pattern(image, label, model)
        
        adv_x = image + eps*perturbations
        adv_x = tf.clip_by_value(adv_x, -1, 1)

        heatmap_adv = cam.compute_heatmap(adv_x)
        heatmap_adv = cv2.resize(heatmap_adv, (image_data.shape[1], image_data.shape[0]))
        (heatmap_adv, output_adv) = cam.overlay_heatmap(heatmap_adv, image_data, alpha=opacity)
        (_, labeladv, confidenceadv) = decode_predictions(model.predict(adv_x), top=1)[0][0]
        attacked = np.squeeze(adv_x*0.5+0.5)
        attacked = cv2.resize(attacked, (image_data.shape[1], image_data.shape[0]))
        ptr = np.squeeze(perturbations*0.5+0.5)
        ptr = cv2.resize(ptr, (image_data.shape[1], image_data.shape[0]))
        return (heatmap, output, labelClass, confidence, heatmap_adv, output_adv, labeladv, confidenceadv, attacked, ptr)

def app(modelName):

    st.set_option('deprecation.showfileUploaderEncoding', False)

    @st.cache(allow_output_mutation=True)
    def load_model(modelName):
        if(modelName == "ResNet50"):
            pretrained_model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', pooling='avg')
        elif(modelName == "MobileNetV2"):
            pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet', pooling='avg')
        elif(modelName == "VGG16"):
            pretrained_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet', pooling='avg')
            
        decode_predictions = tf.keras.applications.imagenet_utils.decode_predictions
        pretrained_model.trainable = False
        return (pretrained_model, decode_predictions)

    (model, decode_predictions) = load_model(modelName)

    st.write(""" #### Fast Gradient Sign Method. \n Loaded model: """, modelName, " \n Weights: ImageNet")
    file = st.file_uploader("Please upload an image", type=["jpg"])

    if file is None:
        pass
    else:
        image = Image.open(file)
        opacity = st.slider("Overlay opacity:", min_value=0.0, max_value=1.0, step=0.01, value=0.7)
        epsilon = st.slider("Choose epsilon value: ", min_value=0.0, max_value=1.0, step=0.01, value=0.01)
        (origCAN, origCANoverlay, origLabel, origConf, advCan, advCANoverlay, advLabel, advConf, attacked, perturbation) = Analyzer.processAttack(image, np.array(image), model, epsilon, opacity, modelName, decode_predictions)
        conf = "{:.2f}".format(origConf * 100)
        confadv = "{:.2f}".format(advConf * 100)

        my_expander = st.expander(label='View attack composition')
        my_expander.write('Fast Gradient Sign Method attack composition')
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

        col4.image(origCAN)
        col4.image(origCANoverlay, caption=origLabel+" "+str(conf)+"%")
        col5.image(advCan)
        col5.image(advCANoverlay, caption=advLabel+" "+str(confadv)+"%")
