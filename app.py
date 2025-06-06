import streamlit as st

st.set_page_config(page_title="My Embedded App", layout="centered")

st.title("Hello from Streamlit!")
st.write("This Streamlit app is running on Render and iframe-embeddable in WordPress.")

st.success("✅ You can now embed this app anywhere using an <iframe>.")
