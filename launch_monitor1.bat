@echo off
cd /d "%~dp0"
python main.py --fullscreen --monitor 1 --scale 2 --load %*
