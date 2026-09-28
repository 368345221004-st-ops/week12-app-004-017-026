# -*- coding: utf-8 -*-
"""Streamlit app — Conversion Predictor (ใบงานสัปดาห์ที่ 12 · ทีมจี้จ่อเจี๊ยบ 004-017-026)

UI เวอร์ชันมืออาชีพ: hero header + ชื่อสมาชิกทีม, การ์ดผลลัพธ์, แท็บข้อมูลโมเดล/วิธีใช้

โมเดล: DecisionTreeClassifier(max_depth=5, random_state=42) → team_model.joblib
ฟีเจอร์: ProductRelated_Duration, PageValues (ตาม best_features.json สัปดาห์ที่ 10)
ข้อมูลทีมถูก MinMax สเกล 0–1 ตั้งแต่สัปดาห์ที่ 3 → แอปสเกล input ก่อนทำนายทุกครั้ง
"""
from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Conversion Predictor · ทีมจี้จ่อเจี๊ยบ",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
    font-family: 'IBM Plex Sans Thai', 'Sarabun', system-ui, sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1B1440 0%, #3C2A78 55%, #6D3B9E 100%);
    border-radius: 22px; padding: 30px 34px 26px; color: #fff;
    box-shadow: 0 24px 48px -28px rgba(27, 20, 64, .78);
    position: relative; overflow: hidden;
}
.hero:after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,255,255,.18), transparent 65%); border-radius: 50%;
}
.hero .kicker { font-size: .82rem; letter-spacing: 2.4px; text-transform: uppercase; opacity: .8; margin-bottom: 6px; }
.hero h1 { font-size: 2.05rem; font-weight: 700; margin: 0 0 8px; line-height: 1.25; }
.hero p { margin: 0; opacity: .92; font-size: .98rem; }
.chips { margin-top: 16px; }
.chip {
    display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.30);
    color: #fff; padding: 6px 14px; border-radius: 999px; font-size: .84rem; margin: 0 8px 8px 0;
    backdrop-filter: blur(4px);
}
.chip.solid { background: #FFFFFF; color: #1B1440; border-color: #FFFFFF; font-weight: 600; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important; border: 1px solid #E6E3F0 !important;
    background: #FFFFFF !important; box-shadow: 0 12px 30px -24px rgba(27, 20, 64, .5);
    padding: 6px 4px;
}
div.stFormSubmitButton > button, div.stButton > button {
    background: linear-gradient(135deg, #3C2A78, #6D3B9E) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: .62rem 1rem !important;
    box-shadow: 0 10px 24px -16px rgba(60, 42, 120, .9);
    transition: transform .12s ease, box-shadow .12s ease;
}
div.stFormSubmitButton > button:hover, div.stButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 14px 26px -14px rgba(60, 42, 120, .95);
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #F8F7FC, #F0EDF8);
    border: 1px solid #E6E3F0; border-radius: 14px; padding: 14px 18px;
}
/* ล็อกสีตัวเลข/ป้าย กันธีม Dark ของผู้ชมทำให้ตัวเลขกลืนกับพื้นการ์ด */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] > div { color: #1B1440 !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #5A5470 !important; }
[data-testid="stMetricDelta"] { color: #1B6B42 !important; }

/* บังคับพื้นแอปเป็นโทนสว่างเสมอ + ให้คอนโทรลของเบราว์เซอร์ใช้โหมดสว่าง */
[data-testid="stAppViewContainer"], .stApp { background: #F6F8FC !important; color-scheme: light; }
[data-testid="stHeader"] { background: transparent !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp .stMarkdown, .stApp .stCaption, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
    color: #1B1440;
}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
.stApp small { color: #6B6580 !important; }

.badge { border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }
.badge h3 { margin: 0 0 6px; font-size: 1.18rem; font-weight: 700; }
.badge p { margin: 0; font-size: .95rem; line-height: 1.6; }
.badge.risk { background: #FDECEC; border: 1px solid #F5C6C0; color: #8F2C22; }
.badge.safe { background: #E9F7EF; border: 1px solid #BFE7D0; color: #1B6B42; }
.badge .tag {
    display: inline-block; font-size: .74rem; letter-spacing: 1.4px; text-transform: uppercase;
    font-weight: 700; opacity: .75; margin-bottom: 6px;
}
.mini { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 2px; }
.mini div {
    flex: 1 1 120px; background: #F8F7FC; border: 1px solid #E6E3F0; border-radius: 12px;
    padding: 10px 12px; text-align: center;
}
.mini span { display: block; font-size: .74rem; color: #6B6580; }
.mini strong { font-size: 1.02rem; color: #1B1440; }
.footer { text-align: center; color: #8A86A0; font-size: .84rem; margin-top: 28px; line-height: 1.8; }
.section-note { color: #5A5470; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "team_model.joblib")


@st.cache_data
def load_meta():
    with open(BASE_DIR / "team_features.json", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()
FEATURES = meta["features"]          # ['ProductRelated_Duration', 'PageValues']
SCALE = meta["scaling"]

# ค่าที่ใช้แสดงผลเท่านั้น — ใช้ .get() กันแอปพังถ้าคีย์หายจากไฟล์ JSON
TEST_ACC = float(meta.get("test_accuracy", 0.8943))
CV_ACC = float(meta.get("cv_accuracy", 0.8844))
THRESHOLD = float(meta.get("threshold", 0.5))
BASE_CV = float(meta.get("baseline_all_features_cv", 0.894))
N_ROWS = int(meta.get("n_rows", 12205))

TEAM = [("004", "กิตติธัช เภารัตน์"), ("017", "ณัฐวุฒิ กล้าหาญ"), ("026", "พัชรา ขันทะมาลา")]


def scale_value(feature: str, value: float) -> float:
    """สเกลค่า input ให้ตรงกับตอนเทรน (MinMax 0–1 จากสัปดาห์ที่ 3)"""
    lo, hi = SCALE[feature]["min"], SCALE[feature]["max"]
    return 0.0 if hi <= lo else (value - lo) / (hi - lo)


def build_input(page_values: float, duration: float) -> pd.DataFrame:
    raw = {"ProductRelated_Duration": duration, "PageValues": page_values}
    return pd.DataFrame([{f: scale_value(f, raw[f]) for f in FEATURES}], columns=FEATURES)


st.markdown(f"""
<div class="hero">
  <div class="kicker">ใบงานสัปดาห์ที่ 12 · Deploy</div>
  <h1>🛒 Conversion Predictor</h1>
  <p>ระบบทำนายพฤติกรรมผู้เข้าชมเว็บที่มีแนวโน้ม <b>เกิด Conversion (ซื้อ)</b> — ห่อโมเดลสุดท้ายของทีมเป็นแอปที่ใช้งานได้จริง<br>
     วิชา 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล</p>
  <div class="chips">
    <span class="chip solid">ทีม จี้จ่อเจี๊ยบ · 004-017-026</span>
    {''.join(f'<span class="chip">{code} · {name}</span>' for code, name in TEAM)}
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

tab_predict, tab_model, tab_help = st.tabs(["🎯  ประเมินโอกาสซื้อ", "🧠  ข้อมูลโมเดล", "📘  วิธีใช้ & ข้อจำกัด"])

with tab_predict:
    col_form, col_result = st.columns([1, 1.05], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown("#### ข้อมูลพฤติกรรมผู้เข้าชม")
            st.markdown(
                f'<div class="section-note">ฟอร์มมี {len(FEATURES)} ช่อง ตรงกับฟีเจอร์ที่ GA เลือกไว้ใน '
                f'<code>best_features.json</code> (สัปดาห์ที่ 10) — ไม่ขาด ไม่เกิน</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            with st.form("conversion_form"):
                pv_max = float(SCALE["PageValues"]["max"])
                dur_max = float(SCALE["ProductRelated_Duration"]["max"])
                page_values = st.number_input(
                    "PageValues — มูลค่าเพจที่ผู้ใช้ดู (คะแนน)",
                    min_value=0.0, max_value=pv_max, value=0.0, step=1.0,
                    help=("ค่ามูลค่ารวมของหน้าที่ผู้ใช้เข้าชมก่อนซื้อ (0 = ยังไม่เข้าเพจที่มีมูลค่า) "
                          "เป็นฟีเจอร์ที่สัมพันธ์กับการเกิด Conversion มากที่สุดใน EDA สัปดาห์ที่ 4 (corr 0.493)"),
                )
                duration = st.number_input(
                    "ProductRelated_Duration — เวลาที่อยู่หน้าสินค้า (วินาที)",
                    min_value=0.0, max_value=dur_max, value=0.0, step=10.0,
                    help=("เวลารวม (วินาที) ที่ผู้ใช้อยู่กับหน้าสินค้า ยิ่งนานมักหมายถึงความสนใจมากขึ้น "
                          f"(ค่าสูงสุดในชุดข้อมูล ≈ {dur_max:,.0f} วินาที)"),
                )
                st.write("")
                submitted = st.form_submit_button("ประเมินโอกาสซื้อ", use_container_width=True)
            st.caption(
                "ข้อมูลทีมถูก MinMax สเกล 0–1 ตั้งแต่สัปดาห์ที่ 3 → แอปสเกลค่าที่กรอกให้อัตโนมัติก่อนทำนาย"
            )

    with col_result:
        with st.container(border=True):
            st.markdown("#### ผลการประเมิน")
            if not submitted:
                st.info("กรอกข้อมูลด้านซ้าย แล้วกด **ประเมินโอกาสซื้อ** เพื่อดูผล")
                st.markdown(
                    '<div class="section-note">ระบบจะแสดง % ความน่าจะเป็นเกิด Conversion '
                    'พร้อมข้อเสนอเชิงธุรกิจสำหรับทีมการตลาด</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("กำลังประมวลผลด้วยโมเดลของทีม..."):
                    payload = build_input(page_values, duration)
                    pred = int(model.predict(payload)[0])
                    proba = model.predict_proba(payload)[0]
                    p = float(proba[list(model.classes_).index(1)]) if 1 in list(model.classes_) else float(pred)

                st.metric("โอกาสเกิด Conversion (ซื้อ)", f"{p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))

                if pred == 1:
                    st.markdown("""
                    <div class="badge safe">
                      <div class="tag">ผลการทำนาย</div>
                      <h3>✅ มีแนวโน้มเกิด Conversion</h3>
                      <p><b>สิ่งที่ควรทำ:</b> จัดลำดับความสำคัญของลูกค้ารายนี้ — ยิงโปรโมชัน/โค้ดส่วนลดเฉพาะบุคคล
                      หรือให้เจ้าหน้าที่แชทติดตามทันที เพราะมีแนวโน้มปิดการขายสูง</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="badge risk">
                      <div class="tag">ผลการทำนาย</div>
                      <h3>⚠️ ยังไม่มีแนวโน้มเกิด Conversion</h3>
                      <p><b>สิ่งที่ควรทำ:</b> ยังไม่ต้องยิงโปรโมชันหนัก — เก็บเข้ากลุ่ม remarketing
                      และกระตุ้นด้วยคอนเทนต์/อีเมล follow-up ก่อน แล้วค่อยประเมินใหม่เมื่อพฤติกรรมเปลี่ยน</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="mini">
                  <div><span>แนวโน้ม</span><strong>{'สูง' if p >= THRESHOLD else 'ต่ำ'}</strong></div>
                  <div><span>เกณฑ์ตัดสินใจ</span><strong>{THRESHOLD:.2f}</strong></div>
                  <div><span>ความแม่นโมเดล</span><strong>{TEST_ACC:.4f}</strong></div>
                </div>""", unsafe_allow_html=True)

                with st.expander("ดูค่าที่ส่งเข้าโมเดลจริง (หลังสเกล 0–1) และเหตุผล"):
                    rows = [{
                        "ฟีเจอร์": f,
                        "ค่าดิบที่กรอก": raw,
                        "ค่าหลังสเกล": round(scale_value(f, raw), 4),
                        "min เดิม": SCALE[f]["min"],
                        "max เดิม": SCALE[f]["max"],
                    } for f, raw in (("PageValues", page_values), ("ProductRelated_Duration", duration))]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.caption(
                        "สเกลด้วยสูตร (ค่าที่กรอก − min) ÷ (max − min) โดยใช้ค่า min/max เดิมจากข้อมูลดิบ "
                        "ให้ตรงกับตอนเทรนในสัปดาห์ที่ 3"
                    )
                    if page_values > 0:
                        st.info("PageValues > 0 คือสัญญาณสำคัญที่สุดของโอกาสซื้อ — ผู้ใช้เข้าชมเพจที่มีมูลค่าสูงแล้ว")

with tab_model:
    st.markdown("#### สรุปโมเดลสุดท้ายของทีม")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ชนิดโมเดล", "Decision Tree")
    m2.metric("max_depth", str(model.get_params().get("max_depth")))
    m3.metric("ความแม่น (test)", f"{TEST_ACC:.4f}")
    m4.metric("ความแม่น (5-fold CV)", f"{CV_ACC:.4f}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**ฟีเจอร์ที่ฟอร์มรับ** (ตรงกับ `best_features.json` สัปดาห์ที่ 10)")
            st.code("\n".join(FEATURES), language="text")
            st.caption("GA สัปดาห์ที่ 10 คัดจาก 28 ฟีเจอร์ เหลือ 2 ฟีเจอร์ โดยความแม่นลดลงเพียงเล็กน้อย")
    with right:
        with st.container(border=True):
            st.markdown("**ข้อมูลและข้อกำหนด**")
            st.write(f"- ข้อมูลเทรน: **{N_ROWS:,} แถว** (Online Shoppers Intention · Kaggle)")
            st.write("- สเกลข้อมูล: **MinMax 0–1** (สัปดาห์ที่ 3) → แอปสเกล input อัตโนมัติ")
            st.write("- ผลลัพธ์: `1` = เกิด Conversion · `0` = ไม่เกิด")
            st.write(f"- เกณฑ์ตัดสินใจ: **{THRESHOLD:.2f}** (probability ≥ เกณฑ์ → ทำนายว่าเกิด Conversion)")
            st.write(f"- เทียบ baseline 28 ฟีเจอร์: CV **{BASE_CV:.4f}**")

with tab_help:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### วิธีใช้")
            st.markdown(
                "1. ใส่ **PageValues** (มูลค่าเพจที่ผู้ใช้ดู) — ถ้ายังไม่เข้าเพจมูลค่าสูงให้ใส่ 0\n"
                "2. ใส่ **ProductRelated_Duration** (เวลาที่อยู่หน้าสินค้า หน่วยวินาที)\n"
                "3. กด **ประเมินโอกาสซื้อ** → ดู % ความน่าจะเป็นและข้อเสนอที่ควรทำ\n"
                "4. เปิด **ข้อมูลโมเดล** เพื่อดูฟีเจอร์และความแม่นที่ใช้อ้างอิง"
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### ข้อจำกัดของโมเดล")
            st.markdown(
                "1. **ข้อมูลไม่สมดุล** — Conversion มีเพียง ~15.6% ของข้อมูล ทำให้ accuracy ดูสูง "
                "แต่ precision/recall ของคลาส Conversion ต่ำกว่า → ใช้เป็น *ตัวช่วยจัดลำดับความสำคัญ*\n"
                "2. **ใช้แค่ 2 ฟีเจอร์** — เพื่อความประหยัด (เก็บข้อมูลน้อยลง ต้นทุนต่ำลง)\n"
                "3. **ต้องสเกลก่อนทำนาย** — แอปจัดการให้แล้ว แต่ถ้านำโมเดลไปใช้ที่อื่นต้องสเกลด้วยค่า min/max ชุดเดียวกัน\n"
                "4. เป็นเวอร์ชันขั้นต้น — ควรทดสอบกับผู้ใช้จริงก่อนใช้งานเต็มรูปแบบ"
            )

st.markdown(f"""
<div class="footer">
  จัดทำโดย <b>ทีม จี้จ่อเจี๊ยบ (004-017-026)</b> — {' · '.join(f'{code} {name}' for code, name in TEAM)}<br>
  วิชา 306-23-06 ปัญญาประดิษฐ์เพื่อธุรกิจดิจิทัล · ใบงานสัปดาห์ที่ 12 · ข้อมูล: Online Shoppers Intention (Kaggle)
</div>
""", unsafe_allow_html=True)
