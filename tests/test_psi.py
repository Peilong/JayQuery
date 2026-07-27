"""PSI unit tests — owned by Wyatt Wilfong.

Run: pytest tests/test_psi.py -v

Tests progress from basic cryptographic building blocks (Week 2)
to full protocol correctness (Week 3).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_psi_output(tmp_path, monkeypatch):
    """Keep PSI tests from overwriting the REAL data/processed/ordered_id_list.json.

    psi_client.run_psi() writes its result to a module-level OUTPUT_PATH pointing at
    data/processed/. Without this redirect, simply running `pytest tests/` replaced
    the real 2305-student intersection with the ~200 synthetic IDs these tests use —
    silently, and with no test failure to signal it. Autouse so no test in this module
    can regress it.
    """
    import psi.psi_client as psi_client

    monkeypatch.setattr(psi_client, "OUTPUT_PATH", tmp_path / "ordered_id_list.json")


# ─────────────────────────────────────────────────────────────────────────────
# Week 2 Tests — ECDH utilities
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.psi
def test_ecdh_basic() -> None:
    """generate_keypair() returns valid (private, public) key pair.
    Run after implementing generate_keypair() in Week 2.
    Expected: PASSES after Week 2 Step 2.
    """
    from psi.ecdh_utils import generate_keypair
    private_key, public_key = generate_keypair()

    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurvePrivateKey,
        EllipticCurvePublicKey,
    )
    assert isinstance(private_key, EllipticCurvePrivateKey), (
        "private_key must be EllipticCurvePrivateKey"
    )
    assert isinstance(public_key, EllipticCurvePublicKey), (
        "public_key must be EllipticCurvePublicKey"
    )


@pytest.mark.psi
def test_blind_id_returns_bytes() -> None:
    """blind_id() returns a bytes object for a valid input.
    Expected: PASSES after Week 2 Step 3.
    """
    from psi.ecdh_utils import blind_id, generate_keypair
    private_key, _ = generate_keypair()
    result = blind_id("student_12345", private_key)

    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    assert len(result) > 0, "blind_id() returned empty bytes"


@pytest.mark.psi
def test_blind_id_deterministic() -> None:
    """blind_id() produces the same output for the same inputs.
    Two calls with the same id + key must produce identical bytes.
    Expected: PASSES after Week 2 Step 3.
    """
    from psi.ecdh_utils import blind_id, generate_keypair
    private_key, _ = generate_keypair()
    result1 = blind_id("student_42", private_key)
    result2 = blind_id("student_42", private_key)
    assert result1 == result2, "blind_id() is not deterministic — same inputs must give same output"


@pytest.mark.psi
def test_blind_id_different_ids() -> None:
    """blind_id() produces different outputs for different student IDs.
    Expected: PASSES after Week 2 Step 3.
    """
    from psi.ecdh_utils import blind_id, generate_keypair
    private_key, _ = generate_keypair()
    result_a = blind_id("student_001", private_key)
    result_b = blind_id("student_002", private_key)
    assert result_a != result_b, "Two different IDs must produce different blinded values"


# ─────────────────────────────────────────────────────────────────────────────
# Week 3 Tests — Full PSI Protocol
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.psi
@pytest.mark.slow
def test_psi_correctness() -> None:
    """Full PSI protocol returns exactly the true intersection.

    Setup: 1000 client IDs, 900 server IDs, 800 in common.
    PSI must return exactly the 800 shared IDs.
    Expected: PASSES after Week 3 Steps 5–7.

    NOTE: This test starts a background server process — it may take 5–10 seconds.
    """
    import multiprocessing

    from psi.psi_client import run_psi
    from psi.psi_server import run_server

    # Build ID sets with known intersection
    client_ids = [str(i) for i in range(1000)]       # IDs 0–999
    server_ids  = [str(i) for i in range(200, 1100)] # IDs 200–1099
    true_intersection = set(str(i) for i in range(200, 1000))  # 800 IDs

    TEST_PORT = 59876

    # Start server in a subprocess
    server_proc = multiprocessing.Process(
        target=run_server,
        args=(server_ids, "localhost", TEST_PORT),
    )
    server_proc.start()

    import time
    time.sleep(0.5)  # give server time to bind

    try:
        result = run_psi(client_ids, server_host="localhost", server_port=TEST_PORT)
        result_set = set(result)

        assert result_set == true_intersection, (
            f"PSI correctness FAILED:\n"
            f"  Expected {len(true_intersection)} IDs in intersection\n"
            f"  Got {len(result_set)} IDs\n"
            f"  Missing: {true_intersection - result_set}\n"
            f"  Extra:   {result_set - true_intersection}"
        )
        n_expected = len(true_intersection)
        print(f"\n✅  PSI CORRECT — {len(result)} IDs in intersection (expected {n_expected})")

    finally:
        server_proc.terminate()
        server_proc.join()


@pytest.mark.psi
@pytest.mark.slow
def test_psi_no_false_positives() -> None:
    """PSI must never include IDs that are NOT in the true intersection.
    Expected: PASSES after Week 3.
    """
    import multiprocessing

    from psi.psi_client import run_psi
    from psi.psi_server import run_server

    client_ids = [str(i) for i in range(500)]
    server_ids  = [str(i) for i in range(300, 800)]
    true_intersection = set(str(i) for i in range(300, 500))

    TEST_PORT = 59877
    server_proc = multiprocessing.Process(
        target=run_server, args=(server_ids, "localhost", TEST_PORT)
    )
    server_proc.start()
    import time
    time.sleep(0.5)

    try:
        result = run_psi(client_ids, server_host="localhost", server_port=TEST_PORT)
        false_positives = set(result) - true_intersection
        assert len(false_positives) == 0, (
            f"PSI returned {len(false_positives)} false positives: {false_positives}"
        )
    finally:
        server_proc.terminate()
        server_proc.join()
