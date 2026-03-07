import sys, traceback, io
sys.path.insert(0, r"C:\Users\LENOVO\OneDrive\Desktop\New project\ESI")
log_path = r"C:\Users\LENOVO\OneDrive\Desktop\New project\ESI\run_log.txt"

buf = io.StringIO()
import contextlib

with open(log_path, "w", encoding="utf-8") as logf:
    with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        try:
            from src.main import main
            main()
            logf.write("\n\nSUCCESS\n")
        except Exception as e:
            logf.write(f"\n\nERROR: {e}\n")
            traceback.print_exc(file=logf)
