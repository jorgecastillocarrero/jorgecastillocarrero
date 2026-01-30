# Create Windows Scheduled Task for Financial Data Daily Update
# Runs at 06:01 AM local time (equivalent to 00:01 ET for Spain timezone)

$taskName = "FinancialDataDailyUpdate"
$batPath = "C:\Users\usuario\financial-data-project\run_daily_update.bat"
$workDir = "C:\Users\usuario\financial-data-project"

# Create the action
$action = New-ScheduledTaskAction -Execute $batPath -WorkingDirectory $workDir

# Create daily trigger at 06:01 AM
$trigger = New-ScheduledTaskTrigger -Daily -At "06:01AM"

# Settings
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

# Register the task (will overwrite if exists)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force

Write-Host "Task '$taskName' created successfully!"
Write-Host "Scheduled to run daily at 06:01 AM (local time)"
Write-Host ""
Write-Host "To verify: schtasks /query /tn FinancialDataDailyUpdate"
