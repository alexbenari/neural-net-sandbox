@echo off
setlocal

set INPUT_TYPE=digit1h
set BATCH_SIZE=1024
set LR=0.0003
set OPTIMIZER=adam

for %%E in (100 200 300 400 500 600) do (
  echo Training with epochs=%%E
  python sandbox.py --train --input-type %INPUT_TYPE% --batch-size %BATCH_SIZE% --epochs %%E --lr %LR% --optimizer %OPTIMIZER% --preload --save-model digit1h-adam-1024-0p0003-%%E --run-name digit1h-adam-1024-0p0003-%%E
)

endlocal
