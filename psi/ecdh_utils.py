"""Blinding primitives for the ECDH private set intersection protocol.

Used by psi_client.py and psi_server.py. The construction is the standard
Diffie-Hellman PSI of Pinkas et al.: each party maps an identifier into a
prime-order group, raises it to its own secret scalar, and exchanges the
results. Because scalar multiplication commutes, both parties reach the same
doubly-blinded element for a shared identifier and learn nothing about the
identifiers they do not share.

    blind(id, k)        = k * H(id)
    second_blind(P, k') = k' * P
    => k' * (k * H(id)) == k * (k' * H(id))

Security rests on the discrete log problem in the group: recovering k from
(H(id), k*H(id)) is infeasible.

HISTORY -- read before changing this file. The original implementation did

    blinded = (k * int(sha256(id))) mod group_order

which is multiplication in the SCALAR FIELD, not a group operation. It
commuted, so the protocol returned correct intersections and every functional
test passed, but it offered no privacy whatsoever: one known (id, blinded)
pair recovers k = blinded * H(id)^-1 mod q by modular inversion, and student
identifiers are a small enumerable space. There was no discrete log anywhere
in it, and no elliptic curve either -- the EC keypair it generated was never
used for the blinding. tests/test_psi_security.py replays that attack and
requires it to fail.

References:
  - Pinkas et al. 2018 (CCS) -- Section 2, ECDH-PSI protocol overview
  - RFC 9496 -- ristretto255 group
"""

from __future__ import annotations

import hashlib

from psi.ristretto import (
    GROUP_ELEMENT_BYTES,
    SCALAR_BYTES,
    hash_to_group,
    is_valid_point,
    random_scalar,
    scalar_mult,
    scalar_mult_base,
)

__all__ = [
    "GROUP_ELEMENT_BYTES",
    "SCALAR_BYTES",
    "blind_id",
    "generate_keypair",
    "hash_id",
    "is_valid_point",
    "second_blind",
    "serialize_public_key",
]


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate this party's PSI keypair.

    Returns:
        ``(secret_scalar, public_element)``. The secret scalar is the blinding
        key; the public element is ``scalar * generator``. The two-round PSI
        does not transmit the public element -- it is returned so a party can
        publish a commitment to its key if an audit needs one.
    """
    scalar = random_scalar()
    return scalar, scalar_mult_base(scalar)


def blind_id(id_string: str, private_key: bytes) -> bytes:
    """Map an identifier into the group and blind it with this party's key.

    Args:
        id_string: The student identifier (e.g. "11391").
        private_key: This party's secret scalar from :func:`generate_keypair`.

    Returns:
        The encoded group element ``k * H(id_string)``.

    Raises:
        ValueError: If the key is malformed or the blinding is degenerate.
    """
    return scalar_mult(hash_to_group(id_string.encode("utf-8")), private_key)


def second_blind(blinded_bytes: bytes, private_key: bytes) -> bytes:
    """Apply this party's key to an already-blinded element.

    Args:
        blinded_bytes: An element received from the peer.
        private_key: This party's secret scalar.

    Returns:
        The doubly-blinded element.

    Raises:
        ValueError: If ``blinded_bytes`` is not a valid group element. Peer
            input is untrusted, so a malformed element is rejected rather than
            multiplied.
    """
    return scalar_mult(blinded_bytes, private_key)


def hash_id(id_string: str) -> bytes:
    """SHA-256 of an identifier.

    Retained for the ID-extraction and manifest tooling. NOT the hash used for
    blinding -- that is the domain-separated hash-to-group in
    :mod:`psi.ristretto`, because a bare digest is an integer, not a group
    element, and blinding an integer is what made the original code insecure.

    Args:
        id_string: Student identifier string.

    Returns:
        32-byte SHA-256 digest.
    """
    return hashlib.sha256(id_string.encode("utf-8")).digest()


def serialize_public_key(public_key: bytes) -> bytes:
    """Encode a public element for transmission.

    ristretto255 elements are already canonical 32-byte encodings, so this is
    an identity check kept for call-site compatibility.

    Args:
        public_key: Encoded group element.

    Returns:
        The same bytes.

    Raises:
        ValueError: If it is not a valid group element.
    """
    if not is_valid_point(public_key):
        raise ValueError("not a valid ristretto255 group element")
    return public_key
