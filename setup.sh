mkdir -p ~/.streamlit/
echo "[general]  
email = \"imane6farah@gmail.com\""  > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = true"  >> ~/.streamlit/config.toml
