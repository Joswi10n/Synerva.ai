# init_env.ps1 — load .env into THIS PowerShell session
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $k,$v = $_ -split '=',2
  [Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
}
Write-Host "✅ .env loaded into this terminal."
