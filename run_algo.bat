@echo off
REM AlgoHub NFL: grades last week, retrains weekly, makes this week's picks/props/posts, pushes to the site.
REM Double-click to run now, or let Task Scheduler run it (see README).
cd /d "%~dp0"
if not exist logs mkdir logs
python nfl_algo.py auto >> logs\auto.log 2>&1
echo Done. Details in logs\auto.log
