param(
    [Parameter(Mandatory=$true)]
    [int]$Issue,

    [Parameter(Mandatory=$true)]
    [ValidateSet("Backlog","Ready","In Progress","Review","Done")]
    [string]$Status
)

$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$Branch = git branch --show-current
if ($LASTEXITCODE -ne 0) {
    throw "Unable to identify Git branch."
}

$Local = git rev-parse HEAD
$Remote = git rev-parse "origin/$Branch"

if ($LASTEXITCODE -ne 0 -or $Local -ne $Remote) {
    throw "Local commits are not synchronised with GitHub. Push first."
}

$TrackedChanges = @(git status --porcelain --untracked-files=no)

if ($TrackedChanges.Count -gt 0) {
    throw "Uncommitted tracked changes found. Commit or resolve first."
}

& "$PSScriptRoot\update_project_board.ps1" -Issue $Issue -Status $Status

if (-not $?) {
    throw "Projects board update failed."
}

Write-Host "PASS: Git synchronised and project board status verified."
