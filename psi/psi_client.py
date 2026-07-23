"""PSI Client — aligns student IDs with Silo B without revealing non-intersecting IDs.

Owner: Wyatt Wilfong
Status: COMPLETE

After PSI, writes data/processed/ordered_id_list.json — the intersection of
Silo A and Silo B student IDs in sorted order. Alex's DataLoaders use this file
to guarantee both silos contain the same students in the same row order.

Run:
    python -m psi.psi_client
"""

import json
import os
import socket
from pathlib import Path

from psi.ecdh_utils import blind_id, generate_keypair, second_blind
from psi.psi_framing import recv_blinded_ids, send_blinded_ids

PSI_SERVER_HOST = os.getenv("PSI_SERVER_HOST", "localhost")
PSI_SERVER_PORT = int(os.getenv("PSI_SERVER_PORT", "50051"))

SILO_A_IDS_PATH = Path("data/processed/silo_a_ids.json")
OUTPUT_PATH = Path("data/processed/ordered_id_list.json")


def run_psi(
    local_ids: list[str],
    server_host: str = PSI_SERVER_HOST,
    server_port: int = PSI_SERVER_PORT,
) -> list[str]:
    """Run the PSI protocol and return the sorted intersection of IDs.

    This is the PRIMARY PSI interface. Alex's data preparation scripts call this.

    Args:
        local_ids: The list of student IDs in Silo A.
        server_host: PSI server hostname.
        server_port: PSI server port.

    Returns:
        Sorted list of student IDs present in BOTH Silo A and Silo B.
        Also writes the result to data/processed/ordered_id_list.json.
    """
    client_private, _ = generate_keypair()
    
    blinded_client = [blind_id(oid, client_private) for oid in local_ids]
    blinded_to_original = {b: oid for b, oid in zip(blinded_client, local_ids)}
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_host, server_port))

        send_blinded_ids(client_socket, blinded_client)
        blinded_server = recv_blinded_ids(client_socket)
        doubly_blinded_client = recv_blinded_ids(client_socket)

    doubly_blinded_server = [second_blind(b, client_private) for b in blinded_server]
    
    intersection = sorted(
        blinded_to_original[b]
        for b, db in zip(blinded_client, doubly_blinded_client)
        if db in doubly_blinded_server
    )
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(intersection))
    
    return intersection

if __name__ == "__main__":
    if not SILO_A_IDS_PATH.exists():
        raise FileNotFoundError(
            f"{SILO_A_IDS_PATH} not found.\n"
            "Run  python scripts/extract_ids_for_psi.py  first."
        )
    local_ids = json.loads(SILO_A_IDS_PATH.read_text())
    print(f"[PSI Client] Loaded {len(local_ids):,} Silo A IDs from {SILO_A_IDS_PATH}")
    result = run_psi(local_ids)
    print(f"[PSI Client] Intersection size: {len(result):,}")
    print(f"[PSI Client] Written to {OUTPUT_PATH}")
    print()
    print("[PSI Client] Next step:")
    print("  python scripts/prepare_silos.py")
 
