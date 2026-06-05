"""PSI wire framing helpers — shared by psi_server.py and psi_client.py.

Owner: Wyatt Wilfong

Defines the binary wire format used to transmit lists of blinded IDs over TCP.

Wire format:
    [4-byte big-endian uint32 count N] [N × 32 bytes]

    blind_id() and second_blind() always return exactly 32 bytes, so no
    per-item length prefix is needed — the receiver can slice the payload
    into fixed-size chunks after reading the count.
"""

import socket
import struct

BLIND_ID_SIZE = 32  # blind_id() and second_blind() always return exactly 32 bytes


def _recv_all(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed prematurely")
        data += chunk
    return data


def recv_blinded_ids(conn: socket.socket) -> list[bytes]:
    """Receive a list of 32-byte blinded IDs.

    Wire format: [4B count N][N * 32B]
    """
    count = struct.unpack(">I", _recv_all(conn, 4))[0]
    raw = _recv_all(conn, count * BLIND_ID_SIZE)
    return [raw[i * BLIND_ID_SIZE:(i + 1) * BLIND_ID_SIZE] for i in range(count)]


def send_blinded_ids(conn: socket.socket, blinded_ids: list[bytes]) -> None:
    """Send a list of 32-byte blinded IDs.

    Wire format: [4B count N][N * 32B]
    """
    conn.sendall(struct.pack(">I", len(blinded_ids)) + b"".join(blinded_ids))