@echo off
setlocal

cd /d "%~dp0.."

rem Add your training commands below. One per line; they run sequentially.
rem Example:
rem python sandbox.py --train --input-type digit1h --batch-size 1024 --epochs 100 --lr 0.0003 --optimizer adam --preload --save-model

python sandbox.py --train --input-type digit1h --batch-size 1024 --epochs 500 --lr 0.0003 --optimizer adam --preload --save-model --model-name digit1h_900_900
python sandbox.py --train --input-type digit1h --batch-size 1024 --epochs 500 --lr 0.0003 --optimizer adam --preload --save-model --model-name digit1h_1000000000

endlocal
