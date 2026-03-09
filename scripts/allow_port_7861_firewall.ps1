# 在运行服务的电脑上以管理员身份运行此脚本，放行 7861 端口供局域网访问
# 右键 PowerShell -> 以管理员身份运行，然后执行: .\scripts\allow_port_7861_firewall.ps1

$port = 7861
$ruleName = "PreenCut-7861"

$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "规则已存在，先删除再创建..."
    Remove-NetFirewallRule -DisplayName $ruleName
}

New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow
Write-Host "已添加防火墙入站规则，允许 TCP 端口 $port。请从另一台电脑访问 http://<本机IP>:7861"
Write-Host "若仍无法访问，请检查: 1) 两台电脑是否同一局域网 2) 本机杀毒/安全软件是否拦截"
