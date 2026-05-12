$python = "C:\Projects\Programming\6700\grok\.venv\Scripts\python.exe"
$base   = "C:\Projects\Programming\6700\grok\grok_impl"
$log    = "$base\results\training_all.log"
Set-Location $base

$runs = @(
    @{op="add";      model="mlp";         out="results/mlp_add"},
    @{op="x3xy2y";   model="mlp";         out="results/mlp_x3xy2y"},
    @{op="multiply"; model="mlp";         out="results/mlp_multiply"},
    @{op="x2xyy2";   model="mlp";         out="results/mlp_x2xyy2"},
    @{op="add";      model="transformer"; out="results/tfm_add"},
    @{op="x3xy2y";   model="transformer"; out="results/tfm_x3xy2y"},
    @{op="multiply"; model="transformer"; out="results/tfm_multiply"},
    @{op="x2xyy2";   model="transformer"; out="results/tfm_x2xyy2"}
)

foreach ($r in $runs) {
    $msg = "$(Get-Date -Format 'HH:mm:ss') === Starting $($r.model) / $($r.op) ==="
    Add-Content -Path $log -Value $msg
    Write-Host $msg

    & $python train.py --n_epochs 50000 --log_every 100 --fourier_every 500 `
        --operation $r.op --model_type $r.model --device cuda `
        --out_dir $r.out 2>&1 | Tee-Object -FilePath $log -Append

    $msg2 = "$(Get-Date -Format 'HH:mm:ss') === Done $($r.model) / $($r.op) ==="
    Add-Content -Path $log -Value $msg2
    Write-Host $msg2
}

Add-Content -Path $log -Value "$(Get-Date -Format 'HH:mm:ss') === ALL RUNS COMPLETE ==="
