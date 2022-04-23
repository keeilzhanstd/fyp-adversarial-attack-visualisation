import streamlit as st
import tensorflow as tf
import PIL.Image as Image

def app(mdl):
    st.set_option('deprecation.showfileUploaderEncoding', False)

    @st.cache(allow_output_mutation=True)
    def load_model():
        pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
        pretrained_model.trainable = False
        return pretrained_model

    model=load_model()
    decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

    st.write(""" #### Carlini & Wagner """)
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])

    if file is None:
        pass
    else:
        image = Image.open(file)
        st.image(image=image)