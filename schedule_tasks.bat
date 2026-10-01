@echo off
REM Run this ONCE to schedule the algo: Fri 5:00 PM, Sun 10:30 AM, Tue 10:00 AM (your PC must be on).
set BAT=%~dp0run_algo.bat
schtasks /Create /F /TN "AlgoHub NFL - Friday"  /TR "\"%BAT%\"" /SC WEEKLY /D FRI /ST 17:00
schtasks /Create /F /TN "AlgoHub NFL - Sunday"  /TR "\"%BAT%\"" /SC WEEKLY /D SUN /ST 10:30
schtasks /Create /F /TN "AlgoHub NFL - Tuesday" /TR "\"%BAT%\"" /SC WEEKLY /D TUE /ST 10:00
echo.
echo Scheduled. To remove later: schtasks /Delete /TN "AlgoHub NFL - Friday" /F  (same for Sunday/Tuesday)
pause
