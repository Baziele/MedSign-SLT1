"""Run a command on the Windows box over SSH. Usage: wssh.py "command" [--timeout s]."""
import paramiko
import sys

from wcreds import HOST, USER, PW

cmd = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 60

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PW, timeout=15, banner_timeout=30)
_, out, err = c.exec_command(cmd, timeout=timeout)
rc = out.channel.recv_exit_status()
print(f"--- rc={rc} ---")
print(out.read().decode(errors="replace"))
e = err.read().decode(errors="replace")
if e:
    print("--- stderr ---")
    print(e)
c.close()
