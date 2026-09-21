# -*- coding: utf-8 -*-
"""Streamlit app — Conversion Predictor (ใบงานสัปดาห์ที่ 12 · งานกลุ่มทีมจี้จ่อเจี๊ยบ)

ห่อ "โมเดลสุดท้ายของทีม" (Decision Tree) เป็นแอปเวอร์ชันแรกที่รันในเครื่องได้
โมเดล: DecisionTreeClassifier(max_depth=5, random_state=42)
ฟีเจอร์: ProductRelated_Duration, PageValues (ตาม best_features.json สัปดาห์ 10)
"""
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Conversion Predictor — ทีมจี้จ่อเจี๊ยบ", page_icon="🛒", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "team_model.joblib"
META_PATH = BASE_DIR / "team_features.json"


@st.cache_resource
def load_model():
    """โหลดโมเดลครั้งเดียวแล้ว cache ไว้"""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    """ข้อมูลกำกับโมเดล: รายชื่อฟีเจอร์ + ค่า min/max เดิม (ก่อนสเกล) + ความแม่น"""
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()
FEATURES = meta["features"]          # ['ProductRelated_Duration', 'PageValues']
SCALE = meta["scaling"]

FIELD_INFO = {
    "PageValues": {
        "label": "PageValues — มูลค่าเพจที่ผู้ใช้ดู (คะแนน)",
        "min": 0.0, "max": float(SCALE["PageValues"]["max"]), "default": 0.0, "step": 1.0,
        "help": ("ค่ามูลค่ารวมของหน้าที่ผู้ใช้เข้าชมก่อนซื้อ (0 = ยังไม่เข้าเพจที่มีมูลค่า) "
                 "เป็นฟีเจอร์ที่สัมพันธ์กับการเกิด Conversion มากที่สุดใน EDA สัปดาห์ที่ 4 (corr 0.493)"),
    },
    "ProductRelated_Duration": {
        "label": "ProductRelated_Duration — เวลาที่อยู่หน้าสินค้า (วินาที)",
        "min": 0.0, "max": float(SCALE["ProductRelated_Duration"]["max"]), "default": 0.0, "step": 1.0,
        "help": ("เวลารวม (วินาที) ที่ผู้ใช้อยู่กับหน้าสินค้า ยิ่งนานมักหมายถึงความสนใจมากขึ้น "
                 "ค่าสูงสุดในชุดข้อมูล ≈ 63,974 วินาที (~17.8 ชั่วโมง)"),
    },
}


def scale_value(feature: str, value: float) -> float:
    """สเกลค่า input ให้ตรงกับตอนเทรน (MinMax 0–1 จากสัปดาห์ที่ 3)"""
    lo = SCALE[feature]["min"]
    hi = SCALE[feature]["max"]
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def build_input(page_values: float, duration: float) -> pd.DataFrame:
    """สร้าง DataFrame 1 แถว เรียงคอลัมน์ตาม FEATURES เป๊ะ + สเกลค่าก่อนส่งโมเดล"""
    raw_values = {"PageValues": page_values, "ProductRelated_Duration": duration}
    row = {f: scale_value(f, raw_values[f]) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES)


st.title("🛒 Conversion Predictor — ทีมจี้จ่อเจี๊ยบ")
st.caption(
    "ใบงานสัปดาห์ที่ 12 · 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · "
    f"โมเดลสุดท้ายของทีม: `{meta['model']}`"
)
st.write(
    "กรอกพฤติกรรมของผู้เข้าชมเว็บ แล้วกด **ทำนายผล** — "
    "ระบบจะประเมินว่า *มีแนวโน้มเกิด Conversion (ซื้อ)* หรือไม่ และเสนอสิ่งที่ควรทำต่อ"
)

left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.form("predict_form"):
        st.subheader("ข้อมูลพฤติกรรมผู้เข้าชม")
        page_values = st.number_input(
            FIELD_INFO["PageValues"]["label"],
            min_value=FIELD_INFO["PageValues"]["min"],
            max_value=FIELD_INFO["PageValues"]["max"],
            value=FIELD_INFO["PageValues"]["default"],
            step=FIELD_INFO["PageValues"]["step"],
            help=FIELD_INFO["PageValues"]["help"],
        )
        duration = st.number_input(
            FIELD_INFO["ProductRelated_Duration"]["label"],
            min_value=FIELD_INFO["ProductRelated_Duration"]["min"],
            max_value=FIELD_INFO["ProductRelated_Duration"]["max"],
            value=FIELD_INFO["ProductRelated_Duration"]["default"],
            step=FIELD_INFO["ProductRelated_Duration"]["step"],
            help=FIELD_INFO["ProductRelated_Duration"]["help"],
        )
        submitted = st.form_submit_button("ทำนายผล", use_container_width=True)
        st.caption(
            "หมายเหตุ: ฟอร์มมี 2 ช่อง ตรงกับฟีเจอร์ที่ GA เลือกไว้ใน `best_features.json` (สัปดาห์ 10) "
            "ไม่ขาดไม่เกิน — โมเดลใช้แค่ 2 ฟีเจอร์นี้"
        )

with right:
    st.subheader("ผลการประเมิน")
    if not submitted:
        st.info("กรอกข้อมูลด้านซ้าย แล้วกด **ทำนายผล** เพื่อดูผล")
    else:
        payload = build_input(page_values, duration)
        pred = int(model.predict(payload)[0])
        proba = model.predict_proba(payload)[0]
        p_conv = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

        st.metric("โอกาสเกิด Conversion (ซื้อ)", f"{p_conv:.1%}")
        st.progress(min(max(p_conv, 0.0), 1.0))

        if pred == 1:
            st.success(
                f"### ✅ มีแนวโน้ม **เกิด Conversion** ({p_conv:.1%})\n"
                "**สิ่งที่ควรทำ:** จัดลำดับความสำคัญของลูกค้ารายนี้ — ยิงโปรโมชัน/โค้ดส่วนลดเฉพาะบุคคล "
                "หรือให้เจ้าหน้าที่แชทติดตามทันที เพราะมีแนวโน้มปิดการขายสูง"
            )
        else:
            st.error(
                f"### ⚠️ ยัง**ไม่มีแนวโน้มเกิด Conversion** ({p_conv:.1%})\n"
                "**สิ่งที่ควรทำ:** ยังไม่ต้องยิงโปรโมชันหนัก — เก็บเข้ากลุ่ม remarketing "
                "และกระตุ้นด้วยคอนเทนต์/อีเมลFollow-up ก่อน แล้วค่อยประเมินใหม่เมื่อพฤติกรรมเปลี่ยน"
            )

        with st.expander("ดูค่าที่ส่งเข้าโมเดลจริง (หลังสเกล 0–1) และข้อมูลฟีเจอร์"):
            show = payload.copy()
            show.insert(0, "ค่าดิบที่กรอก", [f"PageValues={page_values:,.1f} · Duration={duration:,.1f}"])
            st.dataframe(show, use_container_width=True)
            rows = [{
                "ฟีเจอร์": f,
                "ค่าดิบ": raw,
                "สเกล 0–1": round(scale_value(f, raw), 4),
                "min เดิม": SCALE[f]["min"],
                "max เดิม": SCALE[f]["max"],
            } for f, raw in (("PageValues", page_values), ("ProductRelated_Duration", duration))]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                "สเกลด้วยสูตร (ค่าที่กรอก − min) ÷ (max − min) โดยใช้ค่า min/max เดิมจากข้อมูลดิบ "
                "ให้ตรงกับตอนเทรนในสัปดาห์ที่ 3"
            )

with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลของทีม")
    st.write(f"**ทีม:** {meta['team']}")
    st.write(f"**โมเดลสุดท้าย:** `{meta['model']}`")
    st.write(f"**ความแม่น (test):** {meta['test_accuracy']:.4f}")
    st.write(f"**ความแม่น (5-fold CV):** {meta['cv_accuracy']:.4f}")
    st.write(f"**ข้อมูลที่ใช้เทรน:** {meta['n_rows']:,} แถว")
    st.write(f"**ฟีเจอร์ที่ฟอร์มรับ ({len(FEATURES)}):**")
    st.code("\n".join(FEATURES), language="text")
    st.caption(
        "ผลลัพธ์: 1 = เกิด Conversion (ลูกค้าซื้อ) · 0 = ไม่เกิด\n\n"
        "เกณฑ์ตัดสินใจ: ความน่าจะเป็น ≥ 0.5\n\n"
        "โมเดลนี้เป็นเวอร์ชันแรกสำหรับรันในเครื่อง — สัปดาห์หน้าจะนำขึ้น cloud"
    )
