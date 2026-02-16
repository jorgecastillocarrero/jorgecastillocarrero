# Auto Backup Script - Guarda cambios cada 5 minutos
# Ejecutar: powershell -ExecutionPolicy Bypass -File auto_backup.ps1

$projectPath = "C:\Users\usuario\financial-data-project"
$intervalMinutes = 5

Write-Host "=== Auto Backup Iniciado ===" -ForegroundColor Green
Write-Host "Proyecto: $projectPath"
Write-Host "Intervalo: cada $intervalMinutes minutos"
Write-Host "Presiona Ctrl+C para detener"
Write-Host ""

Set-Location $projectPath

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # Verificar si hay cambios
    $status = git status --porcelain

    if ($status) {
        $changedFiles = ($status | Measure-Object -Line).Lines
        Write-Host "[$timestamp] Detectados $changedFiles archivos modificados" -ForegroundColor Yellow

        # Agregar todos los cambios
        git add -A

        # Crear commit con timestamp
        $commitMsg = "Auto-backup $timestamp"
        git commit -m $commitMsg --quiet

        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$timestamp] Backup guardado: $commitMsg" -ForegroundColor Green
        }
    } else {
        Write-Host "[$timestamp] Sin cambios" -ForegroundColor Gray
    }

    # Esperar 5 minutos
    Start-Sleep -Seconds ($intervalMinutes * 60)
}
