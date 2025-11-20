import streamlit as st
from model import predict_risk, get_feature_importance, get_gamified_level, get_lifestyle_quests

st.set_page_config(page_title="FitLevel AI", page_icon="⚡", layout="wide")

st.title("⚡ FitLevel AI – Diabetes Risk Game")
st.caption("Explore your health risk in a gamified way — NOT medical advice")

slider_values = {}
cols = st.columns(3)


with cols[0]:
    slider_values["age"] = st.slider("Age index", 0, 100, 40)
    slider_values["bmi"] = st.slider("Fitness (BMI)", 0, 100, 50)
with cols[1]:
    slider_values["bp"] = st.slider("Blood Pressure", 0, 100, 50)
    slider_values["s1"] = st.slider("Cholesterol", 0, 100, 50)
with cols[2]:
    slider_values["s5"] = st.slider("Triglycerides", 0, 100, 50)
    slider_values["s6"] = st.slider("Blood Sugar", 0, 100, 50)

if st.button("🎮 Check My Health Level!"):
    _, risk = predict_risk(slider_values)
    lvl = get_gamified_level(risk)
    quests = get_lifestyle_quests(slider_values)
    imp = get_feature_importance()

    st.metric("🔥 Risk Score", f"{risk}/100")
    st.write(f"### Level: **{lvl['title']}**")
    st.progress(int(risk))
    st.write(lvl["message"])

    st.divider()
    st.write("### 🎯 Quests for Next Week")
    for q in quests:
        st.markdown(f"- {q}")

    st.divider()
    st.write("### 📌 Top Impact Features")
    st.table({
        "Feature": [x[0] for x in imp[:6]],
        "Importance": [round(x[1], 3) for x in imp[:6]]
    })
else:
    st.info("Adjust sliders → Click **Check My Health Level!**")
