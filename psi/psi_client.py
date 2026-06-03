"""PSI Client — aligns student IDs with Silo B without revealing non-intersecting IDs.

Owner: Wyatt Wilfong
Status: SCAFFOLD — implement TODO sections in Week 3.

After PSI, writes data/processed/ordered_id_list.json — the intersection of
Silo A and Silo B student IDs in sorted order. Alex's DataLoaders use this file
to guarantee both silos contain the same students in the same row order.

Run:
    python -m psi.psi_client
"""

import json
import os
from pathlib import Path
import socket

from psi.ecdh_utils import blind_id, generate_keypair

PSI_SERVER_HOST = os.getenv("PSI_SERVER_HOST", "localhost")
PSI_SERVER_PORT = int(os.getenv("PSI_SERVER_PORT", "50051"))
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

    TODO (Wyatt, Week 3 — Step 6 in CLAUDE.md):
        1. Generate client keypair.
        2. Blind local_ids with client_private_key.
        3. Connect to server via TCP socket.
        4. Send blinded_client_ids as length-prefixed JSON.
        5. Receive (blinded_server_set, doubly_blinded_client_set) from server.
        6. Blind blinded_server_set with client_private_key → doubly_blinded_server.
        7. Intersection = {id for id in local_ids
               if blind_id(id)^both_keys in doubly_blinded_server}
           IMPORTANT: Sort the result before returning.
        8. Write sorted intersection to OUTPUT_PATH as JSON.
        9. Return sorted intersection list.

    Common mistake (see CLAUDE.md):
        Always sort() both sets before comparing — Python sets are unordered.
    """
    client_private, client_public = generate_keypair()
    blinded_client_set = [blind_id(id, client_private) for id in local_ids]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_host, server_port))
        send_data = b"".join(blinded_client_set)  # TODO: convert to length-prefixed JSON
        client_socket.sendall(send_data)

        data = client_socket.recv(64)  # TODO: receive length-prefixed JSON response
        blinded_server_set, doubly_blinded_client_set = json.loads(data)
        doubly_blinded_server = [blind_id(id, client_private) for id in blinded_server_set]
        sorted_client = sorted(doubly_blinded_client_set)
        sorted_server = sorted(doubly_blinded_server)
        intersection = sorted(
            id
            for id in doubly_blinded_client_set
            if blind_id(id, sorted_client) in doubly_blinded_server
        )
        with open(OUTPUT_PATH, "w") as f:
            json.dump(intersection, f)
        client_socket.close()
    return intersection
    
    raise NotImplementedError("TODO (Wyatt, Week 3): implement run_psi()")


if __name__ == "__main__":
    # Demo run with synthetic IDs
    client_ids = [str(i) for i in range(1200)]  # 200 extra IDs not in server
    result = run_psi(client_ids)
    print(f"[PSI Client] Intersection size: {len(result)}")
    print(f"[PSI Client] Written to {OUTPUT_PATH}")
