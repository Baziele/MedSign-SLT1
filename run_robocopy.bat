@echo off
cd /d "%USERPROFILE%\Documents\Code\medsign-multiview"
robocopy data E:\mv-data /E /Z /R:2 /W:3 /NP /NDL /NJH /BYTES /LOG+:robocopy_E2.log
