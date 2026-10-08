#!/usr/bin/env python3
"""Autonomous babysitter for the Windows box (DESKTOP-65SS7MN).

Polls every INTERVAL seconds over SSH (wssh.py): checks boot time, tasklist,
npz count and E: file count. Relaunches whatever died using the orphan-into-
Session-0 technique. Exits only when BOTH targets are met. Logs to babysit.log.
Resilient: SSH failures (mid-reboot) are retried, never fatal.
"""
import subprocess
import time
import datetime
import re
import os

HERE = "/Users/mazen/dev/projects/medsign"
PY = os.path.join(HERE, ".venv312/bin/python")
WSSH = os.path.join(HERE, "wssh.py")
LOG = os.path.join(HERE, "..", "..", "logs", "babysit.log")
PROJ = "Documents\\Code\\medsign-multiview"
# Proven launcher: `start /b ... & ping` orphans into Session 0 (verified:
# runs start and process videos; PowerShell Start-Process silently never starts).
WEXTRACT_CMD = (".venv312w\\Scripts\\python -u wextract.py --max-videos 0 "
                "--workers 8 --min-sid 334 --max-sid 500 >> wextract_25k.log 2>&1")
ROBO_CMD = ("robocopy data E:\\mv-data /E /Z /R:2 /W:3 /NP /NDL /NJH "
            "/BYTES /LOG+:robocopy_E2.log")
TARGET_NPZ = 25050
TARGET_E = 75001
# PAUSED 2026-09-19: extraction off, E: copy only. Set True to resume
# extraction babysitting (launch code below is untouched).
# RESUMED 2026-09-19 ~10:35: E: copy hit 75,001 files, extraction back on.
# PAUSED again 2026-09-19: user stopped extraction.
# RESUMED 2026-09-19 ~16:35: user ordered full extraction, 8 workers.
WATCH_EXTRACTION = False  # PAUSED 2026-09-20: remainder moved to Mac (skip-list)
# OFF 2026-09-19 ~17:25: user is hand-copying the drive on his MacBook.
# No copy ops from here. Set True to resume robocopy babysitting.
WATCH_ROBOCOPY = False
INTERVAL = 120          # 2 min between polls
RETRY_SHORT = 300       # 5 min retry when SSH fails (box may be rebooting)

POLL_CMD = (
    f"cd {PROJ} & systeminfo | findstr /C:\"System Boot Time\" "
    "& tasklist /FI \"IMAGENAME eq python.exe\" "
    "& tasklist /FI \"IMAGENAME eq robocopy.exe\" "
    "& .venv312w\\Scripts\\python -c "
    "\"import os; print('npz:', len(os.listdir('keypoints')) "
    "if os.path.isdir('keypoints') else -1)\" "
    "& .venv312w\\Scripts\\python -c "
    "\"import os; d='E:\\\\mv-data'; "
    "print('Edir:', os.path.isdir(d)); "
    "print('E-files:', sum(len(f) for _,_,f in os.walk(d)) "
    "if os.path.isdir(d) else -1)\" "
    "& powershell -c \"Get-Content robocopy_E3.log -Tail 6\""
)


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wssh(cmd, timeout):
    """Run wssh.py; return combined stdout. Raises on SSH failure/timeout
    so outages are retried, never mistaken for 'processes dead'."""
    p = subprocess.run([PY, WSSH, cmd, str(timeout)],
                       capture_output=True, text=True, timeout=timeout + 30)
    out = (p.stdout or "") + (p.stderr or "")
    if "--- rc=" not in out:
        raise ConnectionError(f"ssh failed, no rc marker: {out[:300]!r}")
    return out


def launch(inner, name):
    full = f"cd {PROJ} & start \"{name}\" /b {inner} & ping -n 300 127.0.0.1 >nul"
    try:
        out = wssh(full, 15)
        log(f"launch {name}: ssh returned early: {out[:150]!r}")
    except Exception as e:
        # Expected: client aborts while ping holds the channel -> Session 0 orphan
        log(f"launch {name}: sent, client aborted as intended ({type(e).__name__})")


def poll():
    return wssh(POLL_CMD, 100)


def main():
    log("babysit daemon started: targets npz>=25050, E-files>=75001")
    last_boot = None
    e_done_logged = False
    npz_done_logged = False
    while True:
        cycle_start = time.time()
        try:
            out = poll()
        except Exception as e:
            log(f"POLL FAILED (ssh {type(e).__name__}); box may be rebooting. "
                f"retry in {RETRY_SHORT}s")
            time.sleep(RETRY_SHORT)
            continue

        low = out.lower()
        m = re.search(r"System Boot Time:\s*(.+)", out)
        boot = m.group(1).strip() if m else "unknown"
        if last_boot is None:
            last_boot = boot
            log(f"boot baseline: {boot}")
        elif boot != last_boot:
            log(f"REBOOT DETECTED: {last_boot} -> {boot}")
            last_boot = boot

        # tasklist prints an INFO line per empty section; a live section
        # lists its image name. Presence of the name == alive.
        py_alive = "python.exe" in low
        robo_alive = "robocopy.exe" in low

        npz = re.search(r"npz:\s*(-?\d+)", out)
        efiles = re.search(r"E-files:\s*(-?\d+)", out)
        edir = re.search(r"Edir:\s*(\w+)", out)
        npz_n = int(npz.group(1)) if npz else None
        e_n = int(efiles.group(1)) if efiles else None

        if WATCH_ROBOCOPY:
            if edir and edir.group(1).lower() != "true":
                log("ALARM: E:\\mv-data NOT VISIBLE - cable may be bumped. Not touching software.")
            elif e_n == -1 or e_n is None:
                log("ALARM: E: file count failed - drive may be gone. Manual check needed.")

        status = (f"poll: boot={boot} py={'ALIVE' if py_alive else 'DEAD'} "
                  f"robo={'ALIVE' if robo_alive else 'DEAD'} npz={npz_n} E={e_n}")
        log(status)

        if not py_alive and WATCH_EXTRACTION:
            log("python DEAD -> relaunching extraction (8 workers)")
            launch(WEXTRACT_CMD, "wextract8")
        elif not py_alive:
            log("poll: extraction paused by user, not relaunching")
        if not robo_alive and WATCH_ROBOCOPY:
            log("robocopy DEAD -> relaunching E: copy")
            launch(ROBO_CMD, "robocopy")
        elif not robo_alive:
            log("poll: robocopy paused by user, not relaunching")

        if e_n is not None and e_n >= TARGET_E and not e_done_logged:
            log(f"MILESTONE: E: copy COMPLETE at {e_n} files (target {TARGET_E})")
            e_done_logged = True
        if not e_done_logged:
            # Authoritative check: robocopy's own final summary (os.walk
            # undercounts due to denied dirs). Total=75001, FAILED=0.
            m = re.search(r"Files\s*:\s*75001\s+\d+\s+75001\s+\d+\s+0\s", out)
            if m:
                log("MILESTONE: E: copy COMPLETE per robocopy summary "
                    "(75001 files, 0 failed)")
                e_done_logged = True
        if npz_n is not None and npz_n >= TARGET_NPZ and not npz_done_logged:
            log(f"MILESTONE: extraction COMPLETE at {npz_n} npz (target {TARGET_NPZ})")
            npz_done_logged = True
        if e_done_logged and (npz_done_logged or not WATCH_EXTRACTION):
            log("WATCHED TARGETS MET - daemon exiting. "
                "Set WATCH_EXTRACTION=True to resume extraction watch.")
            return

        elapsed = time.time() - cycle_start
        time.sleep(max(60, INTERVAL - elapsed))


if __name__ == "__main__":
    main()
