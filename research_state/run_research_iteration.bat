@echo off
REM AMM Autonomous Research Runner
REM Called by Windows Task Scheduler every 30 minutes.
REM Reads research state and continues from where previous iteration left off.

set TIMESTAMP=%date:~-4%-%date:~4,2%-%date:~7,2%_%time:~0,2%-%time:~3,2%
set LOG_DIR=C:\projects\experiments\ai-research\research_state\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%TIMESTAMP%] Starting research iteration >> "%LOG_DIR%\scheduler.log"

claude --dangerously-skip-permissions -p "You are continuing autonomous AI research on Adaptive Modular Memory (AMM). Read the following files FIRST to orient yourself, then continue from where the previous iteration left off: (1) C:\Users\brad\.claude\projects\C--projects-experiments-ai-research\memory\MEMORY.md (2) C:\projects\experiments\ai-research\research_state\sprint02_status.md (3) C:\projects\experiments\ai-research\research_state\latest_results.md if it exists. Execute the next pending step listed in sprint02_status.md. After completing work: update sprint02_status.md marking what was completed and adding new next steps, update MEMORY.md with any new findings, write results to research_state/latest_results.md. Keep all files concise and timestamped. Working directory: C:\projects\experiments\ai-research\sprint02\ The goal is a Level 2 breakthrough: AMM outperforms baselines at K>=16 with statistical significance (p<0.05, 5 seeds). Stop and report clearly if breakthrough confirmed." >> "%LOG_DIR%\iteration_%TIMESTAMP%.log" 2>&1

echo [%TIMESTAMP%] Research iteration complete >> "%LOG_DIR%\scheduler.log"
