@echo off
cd /d C:\Users\usuario\financial-data-project
call venv\Scripts\activate
streamlit run web/app.py --server.port 8514
pause
