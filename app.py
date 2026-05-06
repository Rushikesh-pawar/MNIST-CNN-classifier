import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")


@st.cache_resource
def get_model():
    return load_model("mnist_cnn_model.h5")


model = get_model()

st.title("MNIST Digit Classifier")
st.markdown(
    "Draw a digit (0-9) in the box below. A CNN trained on MNIST "
    "(98.65% test accuracy) will predict it in real time."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    if canvas.image_data is not None and canvas.image_data[..., :3].any():
        img = (
            Image.fromarray(canvas.image_data.astype("uint8"))
            .convert("L")
            .resize((28, 28))
        )
        x = np.array(img, dtype="float32") / 255.0
        x = x.reshape(1, 28, 28, 1)
        probs = model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        st.metric("Predicted digit", pred)
        st.caption(f"Confidence: {probs[pred] * 100:.1f}%")
        st.bar_chart({"probability": probs})
    else:
        st.info("Draw a digit on the canvas to see a prediction.")
