# Create Windows Scheduled Task for Financial Data Daily Update
# Runs at 00:01 AM local time (Spain timezone)
#
# IMPORTANTE: Esta tarea debe ejecutarse sin interrupciones aunque el usuario
# esté usando el ordenador. La descarga de ~5800 símbolos toma ~1.5-2 horas.

$taskName = "FinancialDataDailyUpdate"
$batPath = "C:\Users\usuario\financial-data-project\run_daily_update.bat"
$workDir = "C:\Users\usuario\financial-data-project"

# Create the action
$action = New-ScheduledTaskAction -Execute $batPath -WorkingDirectory $workDir

# Create daily trigger at 00:01 AM (midnight Spain time)
$trigger = New-ScheduledTaskTrigger -Daily -At "00:01AM"

# Settings - CRÍTICO: No detenerse cuando el usuario usa el PC
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -DontStopOnIdleEnd `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

# Register the task (will overwrite if exists)
# Usando el usuario actual con privilegios elevados
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force

Write-Host "Task '$taskName' created successfully!"
Write-Host "Scheduled to run daily at 00:01 AM (Spain time)"
Write-Host ""
Write-Host "Settings configured:"
Write-Host "  - StartWhenAvailable: Yes (runs if missed)"
Write-Host "  - DontStopOnIdleEnd: Yes (continues even if user is active)"
Write-Host "  - WakeToRun: Yes (wakes PC from sleep)"
Write-Host "  - ExecutionTimeLimit: 4 hours"
Write-Host "  - Battery: Runs on battery power"
Write-Host ""
Write-Host "To verify: schtasks /query /tn FinancialDataDailyUpdate"
