@echo off
echo Iniciando Auto Backup cada 5 minutos...
echo.
powershell -ExecutionPolicy Bypass -WindowStyle Normal -File "%~dp0auto_backup.ps1"
