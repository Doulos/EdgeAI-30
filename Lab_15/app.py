
import streamlit as st

st.title("Simple CoCalc Streamlit App")
st.write("Hello! This app is running on CoCalc.")

name = st.text_input("What is your name?")
if name:
    st.write(f"Nice to meet you, {name}!")
