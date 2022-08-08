import socket
import sys
from contextlib import closing
with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
