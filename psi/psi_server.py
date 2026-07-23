"""PSI Server — receives blinded IDs, re-blinds with server key, returns doubly-blinded set.

Owner: Wyatt Wilfong
Status: COMPLETE

Protocol (Pinkas 2018 §2, simplified ECDH-PSI):
  1. Client sends:  {H(id_i)^client_key  for id_i in client_set}
  2. Server sends:  {H(id_i)^server_key  for id_i in server_set}   (server's own blinded set)
              AND:  {H(id_j)^client_key^server_key  for each received item}  (re-blinded client set)
  3. Client computes intersection by comparing doubly-blinded sets.

Run:
    python -m psi.psi_server          # listens on PSI_SERVER_PORT (default 50051)
"""

import json
import os
import socket
from pathlib import Path

from psi.ecdh_utils import blind_id, generate_keypair, second_blind
from psi.psi_framing import recv_blinded_ids, send_blinded_ids

PSI_SERVER_HOST = os.getenv("PSI_SERVER_HOST", "localhost")
PSI_SERVER_PORT = int(os.getenv("PSI_SERVER_PORT", "50051"))

SILO_B_IDS_PATH = Path("data/processed/silo_b_ids.json")

def run_server(
    server_ids: list[str], host: str = PSI_SERVER_HOST, port: int = PSI_SERVER_PORT
) -> None:
    """Start the PSI server, process one client connection, then exit.

    Args:
        server_ids: The list of student IDs in this server's dataset (Silo B).
        host: Server bind address.
        port: Server listen port.
    """
    server_private, _ = generate_keypair()
    blinded_server_set = [blind_id(id, server_private) for id in server_ids]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[PSI Server] Listening on {host}:{port}...")
        conn, addr = server_socket.accept()
        with conn:
            print(f"[PSI Server] Connected by {addr}")

            client_data = recv_blinded_ids(conn)
            doubly_blinded_client_set = [second_blind(b, server_private) for b in client_data]
                
            send_blinded_ids(conn,blinded_server_set)
            send_blinded_ids(conn,doubly_blinded_client_set)

            print(f"[PSI Server] Done — {len(blinded_server_set)} server IDs, "
                  f"{len(doubly_blinded_client_set)} re-blinded client IDs sent.")

if __name__ == "__main__":
    if not SILO_B_IDS_PATH.exists():
            raise FileNotFoundError(
                f"{SILO_B_IDS_PATH} not found.\n"
                "Run  python scripts/extract_ids_for_psi.py  first."
            )
    server_ids = json.loads(SILO_B_IDS_PATH.read_text())
    print(f"[PSI Server] Loaded {len(server_ids):,} Silo B IDs from {SILO_B_IDS_PATH}")
    run_server(server_ids)