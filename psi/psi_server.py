"""PSI Server — receives blinded IDs, re-blinds with server key, returns doubly-blinded set.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement TODO sections in Week 3.

Protocol (Pinkas 2018 §2, simplified ECDH-PSI):
  1. Client sends:  {H(id_i)^client_key  for id_i in client_set}
  2. Server sends:  {H(id_i)^server_key  for id_i in server_set}   (server's own blinded set)
              AND:  {H(id_j)^client_key^server_key  for each received item}  (re-blinded client set)
  3. Client computes intersection by comparing doubly-blinded sets.

Run:
    python -m psi.psi_server          # listens on PSI_SERVER_PORT (default 50051)
"""

import os
import socket

PSI_SERVER_HOST = os.getenv("PSI_SERVER_HOST", "localhost")
PSI_SERVER_PORT = int(os.getenv("PSI_SERVER_PORT", "50051"))


def _recv_all(conn: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed prematurely")
        data += chunk
    return data


def run_server(
    server_ids: list[str], host: str = PSI_SERVER_HOST, port: int = PSI_SERVER_PORT
) -> None:
    """Start the PSI server, process one client connection, then exit.

    Args:
        server_ids: The list of student IDs in this server's dataset (Silo B).
        host: Server bind address.
        port: Server listen port.

    TODO (Wyatt, Week 3 — Steps 5 in CLAUDE.md):
        1. Generate server keypair: server_private, server_public = generate_keypair()
        2. Blind all server IDs: blinded_server = [blind_id(id, server_private) for id in server_ids]
        3. Accept one TCP connection from the client.
        4. Receive the client's blinded ID list (length-prefixed JSON of hex strings).
        5. Re-blind each received item with server_private to get doubly-blinded client set.
        6. Send back: (blinded_server_set, doubly_blinded_client_set) as JSON.
        7. Close connection.
    """
    raise NotImplementedError("TODO (Wyatt, Week 3): implement run_server()")


if __name__ == "__main__":
    # Demo: load server IDs from data/processed/silo_b.pkl or a test fixture
    print(f"[PSI Server] Listening on {PSI_SERVER_HOST}:{PSI_SERVER_PORT}")
    # TODO: load real server IDs here
    demo_ids = [str(i) for i in range(1000)]
    run_server(demo_ids)
