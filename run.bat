@echo off
REM ── เปิดแอป Titanic Survival Predictor (ใบงานสัปดาห์ที่ 12) ──
cd /d "%~dp0"
echo กำลังเปิดแอป... เบราว์เซอร์จะเปิดที่ http://localhost:8501
echo (ปิดหน้าต่างนี้เพื่อหยุดแอป)
".\venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
