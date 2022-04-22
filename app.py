import streamlit as st
from multiapp import MultiApp
from apps import fgsm, bim, cw

app = MultiApp()

st.markdown("""
### Gradient based attacks visualisation.
""")

app.add_app("FGSM", fgsm.app)
app.add_app("BIM", bim.app)
app.add_app("CW", cw.app)

app.run()
