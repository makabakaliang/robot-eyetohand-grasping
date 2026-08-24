"""Minimal TCP client for the robot's JSON protocol.

Unifies the path between mock_robot.py and the real controller so the GUI
does not branch on hardware presence.
"""
import json
import socket


class RobotClient:
    def __init__(self, host="127.0.0.1", port=9760, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)

    def send(self, payload: dict) -> str:
        if self._sock is None:
            raise RuntimeError("not connected; call connect() first")
        self._sock.sendall(json.dumps(payload).encode("utf-8"))
        return self.recv()

    def recv(self, bufsize=4096) -> str:
        data = self._sock.recv(bufsize)
        return data.decode("utf-8", errors="replace").strip()

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
