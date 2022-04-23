import streamlit as st
from multiapp import MultiApp
from apps import fgsm, bim, pgd

app = MultiApp()

st.markdown("""
### Gradient based attacks visualisation.
""")

app.add_app("FGSM", fgsm.app)
app.add_app("BIM", bim.app)
app.add_app("PGD", pgd.app)

app.run()
