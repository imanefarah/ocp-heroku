import streamlit as st
import app2
import main
import streamlit.components.v1 as components


class MultiApp:


    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar.radio(
        st.sidebar.header(' Navigation')
        app = st.sidebar.selectbox(
            '',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()

st.sidebar.image: st.sidebar.image("https://lh3.googleusercontent.com/proxy/5Np-HoGozXJ76YM03SlHQVlIgn2mrVC2WP-7AxGJBPcMXUjZXbrn8ChJ7-CLTbwDfnl5LydU1IbJXaiUERORxfamoRVM1kTl4bTO09xwJGdORND8nBuYzJ3yH22PG8hJHUnUeYBN0o0", use_column_width=True)

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
