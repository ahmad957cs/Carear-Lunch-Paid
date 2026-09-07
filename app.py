import streamlit as st

st.set_page_config(
    page_title="Steel Industry Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2933/2933245.png",
    width=100
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "",

    [

        "🏠 Home",

        "📂 Dataset",

        "📊 EDA",

        "⚙ Feature Engineering",

        "🤖 Model Training",

        "📈 Evaluation",

        "🔮 Prediction",

        "ℹ About"

    ]

)