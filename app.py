import streamlit as st

st.title("AI-Based Road Damage Detection System")

st.write("Upload a road image to begin analysis.")

uploaded_image = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Uploaded Road Image",
        width="stretch"
    )
