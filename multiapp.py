import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'Attack types',
            self.apps,
            format_func=lambda app: app['title'])
        modelName = st.sidebar.selectbox(
        'Select Model',
        ('MobileNetV2', 'ResNet50', 'VGG16'))
        app['function'](modelName)