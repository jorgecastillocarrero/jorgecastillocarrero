@echo off
title Dashboard Financiero
cd /d C:\Users\usuario\financial-data-project
echo Iniciando Dashboard Financiero...
echo.
echo URL: http://localhost:8501
echo.
streamlit run web/app.py
pause
