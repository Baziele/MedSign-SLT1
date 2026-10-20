import paramiko
from wcreds import USER, PW
t = paramiko.Transport(('172.20.10.11', 22)); t.connect(username=USER, password=PW)
s = paramiko.SFTPClient.from_transport(t)
s.get('Documents/Code/medsign-multiview/npz_ship.zip', 'npz_ship.zip')
s.close(); t.close(); print('fetch complete')
