@echo off
cd /d "%USERPROFILE%\Documents\Code\medsign-multiview"
.venv312w\Scripts\python -u wextract.py --max-videos 0 --workers 6 --min-sid 334 --max-sid 500 >> wextract_25k.log 2>&1
