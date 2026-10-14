param(
    [Parameter(Mandatory=$true)]
    [int]$Issue,

    [Parameter(Mandatory=$true)]
    [ValidateSet("Backlog","Ready","In Progress","Review","Done")]
    [string]$Status,

    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$Owner = "dhandan"
$ProjectNumber = 2

$Project = gh project view $ProjectNumber --owner $Owner --format json |
    ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    throw "Could not read GitHub Project."
}

$Fields = gh project field-list $ProjectNumber --owner $Owner --format json |
    ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    throw "Could not read Project fields."
}

$StatusField = $Fields.fields | Where-Object { $_.name -eq "Status" }
$Option = $StatusField.options | Where-Object { $_.name -eq $Status }

if (-not $Option) {
    throw "Status option not found: $Status"
}

$Board = gh project item-list $ProjectNumber --owner $Owner --limit 100 --format json |
    ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    throw "Could not read board items."
}

$Item = @($Board.items | Where-Object {
    $_.content.number -eq $Issue -and
    $_.content.repository -eq "dhandan/football-copilot"
})

if ($Item.Count -ne 1) {
    throw "Expected exactly one board item for issue #$Issue."
}

$Card = $Item[0]

Write-Host "Issue: #$Issue"
Write-Host "Current status: $($Card.status)"
Write-Host "Requested status: $Status"

if ($CheckOnly) {
    Write-Host "PASS: Read-only board check completed."
    exit 0
}

if ($Card.status -eq $Status) {
    Write-Host "PASS: Board already has the correct status."
    exit 0
}

gh project item-edit `
    --project-id $Project.id `
    --id $Card.id `
    --field-id $StatusField.id `
    --single-select-option-id $Option.id

if ($LASTEXITCODE -ne 0) {
    throw "GitHub board update failed."
}

$Verified = gh project item-list $ProjectNumber --owner $Owner --limit 100 --format json |
    ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify updated board."
}

$Updated = $Verified.items | Where-Object {
    $_.content.number -eq $Issue -and
    $_.content.repository -eq "dhandan/football-copilot"
}

if ($Updated.status -ne $Status) {
    throw "Verification failed. Board status was not updated."
}

Write-Host "PASS: GitHub Projects status verified as $Status"
