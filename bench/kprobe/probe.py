import socket
try:
    import mediapipe
    print("mp preinstalled", mediapipe.__version__)
except ImportError:
    print("no mp")
try:
    print("dns:", socket.gethostbyname("pypi.org"))
except Exception as e:
    print("no net:", str(e)[:60])
