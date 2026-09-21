"""Copy a local file to the Windows box. Usage: wput.py local [remote]."""
import paramiko
import sys

HOST, USER, PW = "172.20.10.11", "baziele", "bazilarate"

local = sys.argv[1]
remote = sys.argv[2] if len(sys.argv) > 2 else "Documents/Code/medsign-multiview/" + local.split("/")[-1]

t = paramiko.Transport((HOST, 22))
t.connect(username=USER, password=PW)
s = paramiko.SFTPClient.from_transport(t)
s.put(local, remote)
s.close()
t.close()
print(f"put {local} -> {remote}")
