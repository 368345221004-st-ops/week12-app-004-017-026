# -*- coding: utf-8 -*-
"""Streamlit app (โหมดงานเดี่ยว ภาค B ขั้น 1–7) — Titanic Survival Predictor

เก็บไว้เป็นหลักฐานของขั้นตอนแรกในเวิร์กชอป (ก่อนสลับเป็นโมเดลทีมในขั้นที่ 8)
โมเดล: DecisionTreeClassifier(max_depth=3, random_state=42)
ฟีเจอร์: Pclass, Sex_female, Age, Fare, FamilySize
รันด้วย: streamlit run app_titanic_stepB.py
"""
from pathlib import Path
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ทำนายการรอดชีวิต Titanic", page_icon="🚢", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "titanic_model.joblib"
FEATURES = ["Pclass", "Sex_female", "Age", "Fare", "FamilySize"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def build_input(pclass: int, is_female: bool, age: float,
                sibsp: int, parch: int, fare: float) -> pd.DataFrame:
    family_size = sibsp + parch + 1
    return pd.DataFrame([{
        "Pclass": int(pclass),
        "Sex_female": 1 if is_female else 0,
        "Age": float(age),
        "Fare": float(fare),
        "FamilySize": int(family_size),
    }], columns=FEATURES)


st.title("🚢 Titanic Survival Predictor")
st.caption("งานเดี่ยว ภาค B ขั้น 1–7 · โมเดล Decision Tree (max_depth=3) — ตัวอย่างฝึกก่อนสลับเป็นโมเดลทีม")

with st.form("predict_form"):
    st.subheader("ข้อมูลผู้โดยสาร")
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3], index=2,
                          format_func=lambda x: {1: "1 — ชั้นหนึ่ง", 2: "2 — ชั้นสอง", 3: "3 — ชั้นสาม"}[x],
                          help="ชั้นโดยสารมีผลต่อโอกาสรอดชีวิตสูงสุดในข้อมูล Titanic")
    sex = st.radio("เพศ", ["หญิง", "ชาย"], horizontal=True, index=1,
                   help="ในข้อมูลจริง ผู้โดยสารหญิงมีอัตรารอดชีวิตสูงกว่าชายอย่างชัดเจน")
    age = st.slider("อายุ (ปี)", 0, 80, 28, 1, help="อายุเป็นปี (ข้อมูลจริงมีช่วง 0.42–80 ปี)")
    c1, c2 = st.columns(2)
    sibsp = c1.number_input("พี่น้อง/คู่สมรสบนเรือ (SibSp)", 0, 8, 0, 1,
                            help="จำนวนพี่น้องหรือคู่สมรสที่เดินทางด้วยกัน")
    parch = c2.number_input("พ่อแม่/ลูกบนเรือ (Parch)", 0, 6, 0, 1,
                            help="จำนวนพ่อแม่หรือลูกที่เดินทางด้วยกัน")
    fare = st.number_input("ค่าโดยสาร (Fare)", 0.0, 512.0, 32.0, 0.5, format="%.2f",
                           help="ค่าโดยสารเป็นปอนด์ (ข้อมูลจริงสูงสุด 512.33)")
    submitted = st.form_submit_button("ทำนายผล", use_container_width=True)

if submitted:
    payload = build_input(pclass, sex == "หญิง", age, sibsp, parch, fare)
    pred = int(model.predict(payload)[0])
    proba = model.predict_proba(payload)[0]
    p_survive = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

    st.divider()
    if pred == 1:
        st.success(f"### ✅ ทำนายว่า **รอดชีวิต** (ความน่าจะเป็น {p_survive:.1%})")
    else:
        st.error(f"### ❌ ทำนายว่า **ไม่รอดชีวิต** (ความน่าจะเป็นรอด {p_survive:.1%})")
    st.progress(min(max(p_survive, 0.0), 1.0), text=f"ความน่าจะเป็นรอดชีวิต: {p_survive:.1%}")

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล (5 ฟีเจอร์)"):
        show = payload.copy()
        show["Sex_female"] = show["Sex_female"].map({1: "1 (หญิง)", 0: "0 (ชาย)"})
        st.dataframe(show, use_container_width=True)
        st.caption(f"FamilySize = SibSp({sibsp}) + Parch({parch}) + 1 = {sibsp + parch + 1}")

with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดล")
    st.write(f"**ชนิด:** `{type(model).__name__}`")
    st.write(f"**max_depth:** `{model.get_params().get('max_depth')}`")
    st.write(f"**ฟีเจอร์ที่ใช้ ({len(FEATURES)}):**")
    st.code("\n".join(FEATURES), language="text")
    st.caption("Sex_female: 1 = หญิง, 0 = ชาย\n\nFamilySize = SibSp + Parch + 1")
