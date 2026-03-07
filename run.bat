@echo off
cd /d "C:\Users\LENOVO\OneDrive\Desktop\New project\ESI"
".venv\Scripts\python.exe" -m pip install matplotlib scikit-learn pandas numpy > pip_install.log 2>&1
".venv\Scripts\python.exe" run_pipeline.py > run_out.txt 2>&1
echo DONE >> run_out.txt
