import streamlit as st
import app2
import main
import codecs
import streamlit.components.v1 as components
from PIL import Image
st.sidebar.image: st.sidebar.image("https://lh3.googleusercontent.com/proxy/PeIGIMPNzsk5ofZkoWlxXaCOp--0X50lGJNstKJ_YkEqX6RwtLuLpRTv5tBjaB2QX8LAIgYQfXk5zliK0Dt3xdJxRmdjlV6mftTu6mqedjZ-BjGbLLfDSvjAKl45FvGEHzUEbERkqUE", use_column_width=True)

primaryColor = st.get_option("theme.primaryColor")
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid #579D02; border-radius:None }}

     }}
</style>
"""
st.markdown(s, unsafe_allow_html=True)
#st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/OCP_Group.svg/1200px-OCP_Group.svg.png",use_column_width= True)
app = MultiApp()

# Add all your application here
app.add_app("Simple User",app2.app)
app.add_app("Data Scientist User",main.app)
# The main app
app.run()
