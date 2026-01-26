@echo off
setlocal

cd /d "%~dp0.."

set EPOCHS=500
set BATCH_SIZE=1024
set LR=0.0003
set OPTIMIZER=adam
set "timestamp=%DATE%_%TIME%"
set "timestamp=%timestamp:/=%"
set "timestamp=%timestamp::=%"
set "timestamp=%timestamp:.=%"
set "timestamp=%timestamp: =0%"



for %%T in (digit digit1h binary normalized-int) do (
  echo Training with format=%%T
  python sandbox.py --train --input-type %%T --batch-size %BATCH_SIZE% --epochs %EPOCHS% --lr %LR% --optimizer %OPTIMIZER% --preload --save-model %%T-adam-1024-0p0003-500-%timestamp% --run-name %timestamp%-%%T-adam-1024-0p0003-500
)

endlocal
