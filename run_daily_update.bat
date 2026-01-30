@echo off
REM Daily Financial Data Update Script
REM Scheduled to run at 00:01 ET via Windows Task Scheduler

cd /d C:\Users\usuario\financial-data-project

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the daily update
py -m src.scheduler --run-now

REM Log completion
echo [%date% %time%] Daily update completed >> scheduler_runs.log
