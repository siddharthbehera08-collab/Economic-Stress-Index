import subprocess, sys, os
os.chdir(r"C:\Users\LENOVO\OneDrive\Desktop\New project\ESI")
result = subprocess.run(
    [r".\.venv\Scripts\python.exe", "-m", "src.main"],
    capture_output=True, text=True,
    cwd=r"C:\Users\LENOVO\OneDrive\Desktop\New project\ESI"
)
with open("pipeline_stdout.txt","w",encoding="utf-8") as f:
    f.write(result.stdout)
with open("pipeline_stderr.txt","w",encoding="utf-8") as f:
    f.write(result.stderr)
print("Done. Exit:", result.returncode)
