"""ECDH utility functions for the PSI protocol.

Owner: Wyatt Wilfong
Status: COMPLETE

This module provides the low-level cryptographic building blocks used by
psi_client.py and psi_server.py. All functions use the SECP256R1 elliptic curve
via the Python `cryptography` library.

References:
  - Pinkas et al. 2018 (CCS) — Section 2 for ECDH-PSI protocol overview
  - cryptography library docs: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
"""

import hashlib

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.ec import (
    ECDH,
    SECP256R1,
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
    derive_private_key,
    generate_private_key,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)


def generate_keypair() -> tuple[EllipticCurvePrivateKey, EllipticCurvePublicKey]:
    """Generate an ECDH key pair on the SECP256R1 (P-256) curve.

    Returns:
        A tuple of (private_key, public_key).

    Example:
        private_key, public_key = generate_keypair()

    TODO (Wyatt, Week 2 — Step 2 in CLAUDE.md):
        Implement this function using generate_private_key(SECP256R1(), default_backend()).
        Return (private_key, private_key.public_key()).
    """
    private_key = generate_private_key(SECP256R1(), default_backend())
    return (private_key, private_key.public_key())


def blind_id(id_string: str, private_key: EllipticCurvePrivateKey) -> bytes:
    """Hash an ID string to a curve point and blind it with the private key.

    This is the core PSI operation: H(id)^private_key where H maps strings to
    curve points via hash-then-derive. We approximate this using ECDH exchange
    with a deterministically derived peer key from the hash of id_string.

    Args:
        id_string: The student identifier to blind (e.g., "12345").
        private_key: The ECDH private key for this party.

    Returns:
        64 bytes representing the blinded ID point (uncompressed x,y coords).

    TODO (Wyatt, Week 2 — Step 3 in CLAUDE.md):
        1. Hash id_string with SHA-256 to get 32 bytes.
        2. Use those bytes as the x-coordinate to derive a deterministic public key.
           (Ask Alex or Dr. Li for the `_derive_public_key_from_hash` helper pattern.)
        3. Perform ECDH exchange: shared = private_key.exchange(ECDH(), derived_pubkey)
        4. Return the shared secret bytes (64 bytes).
    """
    h_id = hash_id(id_string)
    scale = int.from_bytes(h_id, byteorder="big")
    scale = (scale % SECP256R1().group_order - 1) + 1
    derived_privkey = derive_private_key(scale, SECP256R1())
    derived_pubkey = derived_privkey.public_key()
    shared = private_key.exchange(ECDH(), derived_pubkey)
    return shared


def hash_id(id_string: str) -> bytes:
    """SHA-256 hash of an ID string.

    Args:
        id_string: Student identifier string.

    Returns:
        32-byte SHA-256 digest.
    """
    return hashlib.sha256(id_string.encode("utf-8")).digest()


def serialize_public_key(public_key: EllipticCurvePublicKey) -> bytes:
    """Serialize a public key to uncompressed point bytes for transmission.

    Args:
        public_key: ECDH public key.

    Returns:
        65 bytes (uncompressed point: 0x04 || x || y).
    """
    return public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
