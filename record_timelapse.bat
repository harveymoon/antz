@echo off
REM ============================================================
REM  Timelapse recording run
REM  - 2000x2000 world, one PNG frame every 100 sim steps
REM  - Frames land in dataSave\captures\<runID>\frame_000001.png ...
REM  - Extra args pass through, e.g.:  record_timelapse.bat --load --paths
REM
REM  Assemble into a video afterwards (from inside the capture folder):
REM    ffmpeg -framerate 30 -i frame_%%06d.png -c:v libx264 -pix_fmt yuv420p timelapse.mp4
REM ============================================================
cd /d "%~dp0"
python main.py --size 2000x2000 --capture 100 %*
pause
