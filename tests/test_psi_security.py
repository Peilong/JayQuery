"""Adversarial tests for the PSI blinding primitive.

Run: pytest tests/test_psi_security.py -v

tests/test_psi.py checks that the protocol *works*: deterministic, distinct
per ID, correct intersection, no false positives. The original implementation
passed every one of those while providing no privacy at all -- it computed

    blinded = (k * H(id)) mod q

which is multiplication in the scalar field, not a group operation. One known
(id, blinded) pair recovers k = blinded * H(id)^-1 mod q, and from there every
other blinded value unblinds. Functional tests cannot see that; these can.

The central test here (`test_known_pair_does_not_recover_the_key`) replays
that exact attack and requires it to fail.
"""

import hashlib

import pytest

_IDS = ["11391", "28400", "31604", "32885", "38053"]


@pytest.mark.psi
class TestGroupElementOutput:
    """Blinding must produce a group element, not a scalar-field integer."""

    def test_output_is_a_valid_group_element(self) -> None:
        from psi.ecdh_utils import blind_id, generate_keypair, is_valid_point

        priv, _ = generate_keypair()
        blinded = blind_id("11391", priv)
        assert isinstance(blinded, bytes)
        assert is_valid_point(blinded), "blinded value is not a group element"

    def test_output_is_group_sized(self) -> None:
        from psi.ecdh_utils import GROUP_ELEMENT_BYTES, blind_id, generate_keypair

        priv, _ = generate_keypair()
        assert len(blind_id("11391", priv)) == GROUP_ELEMENT_BYTES

    def test_hashing_alone_is_not_the_blinded_value(self) -> None:
        """A blinded ID must not be derivable without the key."""
        from psi.ecdh_utils import blind_id, generate_keypair, hash_id

        priv, _ = generate_keypair()
        assert blind_id("11391", priv) != hash_id("11391")


@pytest.mark.psi
class TestKeyRecoveryResistance:
    """The break that functional tests missed."""

    def test_known_pair_does_not_recover_the_key(self) -> None:
        """THE regression test for the original flaw.

        An adversary who learns one (id, blinded) pair -- trivial here, since
        student IDs are a small enumerable space -- previously recovered the
        secret by modular division and unblinded the entire set. Replay that
        attack and require it to fail.
        """
        from psi.ecdh_utils import blind_id, generate_keypair

        priv, _ = generate_keypair()
        known_id, target_id = _IDS[0], _IDS[1]
        known_blinded = blind_id(known_id, priv)
        target_blinded = blind_id(target_id, priv)

        # The original arithmetic: treat everything as integers mod the group
        # order and solve for k.
        q = 2**252 + 27742317777372353535851937790883648493  # ristretto255 order
        h_known = int.from_bytes(hashlib.sha256(known_id.encode()).digest(), "big") % q
        b_known = int.from_bytes(known_blinded, "little") % q
        recovered_k = (b_known * pow(h_known, -1, q)) % q

        h_target = int.from_bytes(hashlib.sha256(target_id.encode()).digest(), "big") % q
        forged = ((recovered_k * h_target) % q).to_bytes(32, "little")

        assert forged != target_blinded, (
            "a single known (id, blinded) pair recovered the secret and forged "
            "another ID's blinded value — the blinding is linear in the scalar "
            "field and provides no privacy"
        )

    def test_blinding_is_not_scalar_field_multiplication(self) -> None:
        """Structural form of the same check: the output must not equal the
        integer product that the broken implementation computed."""
        from psi.ecdh_utils import blind_id, generate_keypair, hash_id

        priv, _ = generate_keypair()
        q = 2**252 + 27742317777372353535851937790883648493
        k = int.from_bytes(priv, "little") % q
        h = int.from_bytes(hash_id("11391"), "big") % q

        naive = ((k * h) % q).to_bytes(32, "little")
        assert blind_id("11391", priv) != naive

    def test_different_keys_give_unrelated_blindings(self) -> None:
        from psi.ecdh_utils import blind_id, generate_keypair

        a, _ = generate_keypair()
        b, _ = generate_keypair()
        assert blind_id("11391", a) != blind_id("11391", b)

    def test_the_key_is_not_recoverable_from_many_pairs(self) -> None:
        """With several pairs under one key, every candidate ratio must differ
        — a constant ratio across pairs would leak the key directly."""
        from psi.ecdh_utils import blind_id, generate_keypair

        priv, _ = generate_keypair()
        q = 2**252 + 27742317777372353535851937790883648493
        ratios = set()
        for sid in _IDS:
            h = int.from_bytes(hashlib.sha256(sid.encode()).digest(), "big") % q
            b = int.from_bytes(blind_id(sid, priv), "little") % q
            ratios.add((b * pow(h, -1, q)) % q)
        assert len(ratios) == len(_IDS), (
            "the blinded/hash ratio is constant across IDs — that constant IS "
            "the secret key"
        )


@pytest.mark.psi
class TestProtocolAlgebra:
    """Properties the two-round protocol depends on."""

    def test_double_blinding_commutes(self) -> None:
        """Both parties must reach the same value regardless of order, or the
        intersection silently comes back empty."""
        from psi.ecdh_utils import blind_id, generate_keypair, second_blind

        a, _ = generate_keypair()
        b, _ = generate_keypair()
        assert second_blind(blind_id("11391", a), b) == second_blind(
            blind_id("11391", b), a
        )

    def test_double_blinding_still_a_group_element(self) -> None:
        from psi.ecdh_utils import (
            blind_id,
            generate_keypair,
            is_valid_point,
            second_blind,
        )

        a, _ = generate_keypair()
        b, _ = generate_keypair()
        assert is_valid_point(second_blind(blind_id("11391", a), b))

    def test_distinct_ids_stay_distinct_after_double_blinding(self) -> None:
        from psi.ecdh_utils import blind_id, generate_keypair, second_blind

        a, _ = generate_keypair()
        b, _ = generate_keypair()
        out = {second_blind(blind_id(i, a), b) for i in _IDS}
        assert len(out) == len(_IDS), "a collision would create a false positive"

    def test_rejects_a_non_group_element(self) -> None:
        """Garbage from the wire must be rejected, not silently scalar-multiplied."""
        from psi.ecdh_utils import generate_keypair, second_blind

        priv, _ = generate_keypair()
        with pytest.raises(ValueError, match="point|group|element"):
            second_blind(b"\xff" * 32, priv)


@pytest.mark.psi
class TestKeyGeneration:
    """Secret scalars must be random and well-formed."""

    def test_keys_are_distinct_per_call(self) -> None:
        from psi.ecdh_utils import generate_keypair

        assert len({generate_keypair()[0] for _ in range(32)}) == 32

    def test_scalar_is_group_sized(self) -> None:
        from psi.ecdh_utils import SCALAR_BYTES, generate_keypair

        priv, _ = generate_keypair()
        assert len(priv) == SCALAR_BYTES

    def test_public_element_is_valid(self) -> None:
        """The public value is now a real group element, not decoration."""
        from psi.ecdh_utils import generate_keypair, is_valid_point

        _, pub = generate_keypair()
        assert is_valid_point(pub)
