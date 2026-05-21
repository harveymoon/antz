@echo off
echo ============================================================
echo   Antz Data Organizer - Keep Best Per Run
echo ============================================================
echo.
echo This will keep the single best save file from each sim run
echo and DELETE all other JSON and PNG files.
echo.
python organize_data.py
echo.
echo ============================================================
echo   Done! Best files are now in dataSave\best\
echo ============================================================
pause
