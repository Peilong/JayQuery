"""Minimal ctypes binding to libsodium's ristretto255 group.

Why ristretto255 and not a NIST curve via a pure-Python library:

  * It is a **prime-order** group. No cofactor, so no small-subgroup checks to
    forget and no invalid-curve class of bug.
  * libsodium ships a real, constant-time **hash-to-group**
    (``crypto_core_ristretto255_from_hash``). Hand-rolled hash-to-curve by
    try-and-increment is where prototype PSI implementations usually go wrong:
    it is variable-time in the number of attempts, which leaks about the input.
  * Scalar multiplication is constant-time and rejects invalid encodings.

PyNaCl does not expose these bindings (checked 1.6.2), and the libsodium it
bundles does not export the symbols, so this binds the system library
directly.

Install libsodium:
    macOS          brew install libsodium
    Debian/Ubuntu  apt-get install libsodium23       (CI)
                   apt-get install libsodium-dev     (build)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import threading

#: Encoded group element size.
GROUP_ELEMENT_BYTES = 32
#: Secret scalar size.
SCALAR_BYTES = 32
#: Input width required by crypto_core_ristretto255_from_hash.
HASH_BYTES = 64

#: Domain separation, so a hash used here can never collide with one used for
#: another purpose on the same identifier.
_DOMAIN = b"JayQuery-PSI-ristretto255-v1"

_SEARCH_PATHS = (
    "/opt/homebrew/lib/libsodium.dylib",
    "/opt/homebrew/lib/libsodium.26.dylib",
    "/usr/local/lib/libsodium.dylib",
    "/usr/lib/x86_64-linux-gnu/libsodium.so.23",
    "/usr/lib/x86_64-linux-gnu/libsodium.so",
    "/usr/lib/libsodium.so",
)

_REQUIRED = (
    "crypto_core_ristretto255_from_hash",
    "crypto_scalarmult_ristretto255",
    "crypto_scalarmult_ristretto255_base",
    "crypto_core_ristretto255_scalar_random",
    "crypto_core_ristretto255_is_valid_point",
)

_lib: ctypes.CDLL | None = None
_lock = threading.Lock()


class SodiumUnavailableError(RuntimeError):
    """libsodium is missing or too old to provide ristretto255."""


def _load() -> ctypes.CDLL:
    """Locate, validate and initialize libsodium (cached)."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is not None:
            return _lib

        tried: list[str] = []
        found = ctypes.util.find_library("sodium")
        for path in ([found] if found else []) + list(_SEARCH_PATHS):
            tried.append(path)
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            if all(hasattr(lib, sym) for sym in _REQUIRED):
                if lib.sodium_init() < 0:
                    raise SodiumUnavailableError("sodium_init() failed")
                _lib = lib
                return _lib

        raise SodiumUnavailableError(
            "libsodium with ristretto255 support not found. The PSI protocol "
            "needs it for hash-to-group and scalar multiplication.\n"
            "  macOS:         brew install libsodium\n"
            "  Debian/Ubuntu: apt-get install libsodium23\n"
            f"Tried: {', '.join(p for p in tried if p)}"
        )


def available() -> bool:
    """Whether a usable libsodium is present (for skipping tests)."""
    try:
        _load()
    except SodiumUnavailableError:
        return False
    return True


def random_scalar() -> bytes:
    """A uniform non-zero scalar, via libsodium's CSPRNG."""
    buf = ctypes.create_string_buffer(SCALAR_BYTES)
    _load().crypto_core_ristretto255_scalar_random(buf)
    return buf.raw[:SCALAR_BYTES]


def hash_to_group(data: bytes) -> bytes:
    """Map arbitrary bytes to a group element (constant-time).

    Uses SHA-512 over a domain-separated input to fill the 64 bytes
    ``from_hash`` requires. The discrete log of the result with respect to the
    generator is unknown, which is what makes the blinding one-way.

    Args:
        data: Input bytes (an identifier).

    Returns:
        A ``GROUP_ELEMENT_BYTES`` encoded element.
    """
    digest = hashlib.sha512(_DOMAIN + data).digest()
    out = ctypes.create_string_buffer(GROUP_ELEMENT_BYTES)
    if _load().crypto_core_ristretto255_from_hash(out, digest) != 0:
        raise ValueError("ristretto255 from_hash failed")
    return out.raw[:GROUP_ELEMENT_BYTES]


def scalar_mult(point: bytes, scalar: bytes) -> bytes:
    """Multiply a group element by a secret scalar.

    Args:
        point: Encoded group element.
        scalar: Secret scalar.

    Returns:
        The encoded product element.

    Raises:
        ValueError: If the inputs are malformed, the point is not a valid
            group element, or the result is the identity (which would erase
            the input rather than blind it).
    """
    if len(point) != GROUP_ELEMENT_BYTES:
        raise ValueError(
            f"point must be {GROUP_ELEMENT_BYTES} bytes, got {len(point)}"
        )
    if len(scalar) != SCALAR_BYTES:
        raise ValueError(f"scalar must be {SCALAR_BYTES} bytes, got {len(scalar)}")
    out = ctypes.create_string_buffer(GROUP_ELEMENT_BYTES)
    if _load().crypto_scalarmult_ristretto255(out, scalar, point) != 0:
        raise ValueError(
            "ristretto255 scalar multiplication rejected its input: not a valid "
            "group element, or the result was the identity."
        )
    return out.raw[:GROUP_ELEMENT_BYTES]


def scalar_mult_base(scalar: bytes) -> bytes:
    """Multiply the generator by a scalar (the public element)."""
    if len(scalar) != SCALAR_BYTES:
        raise ValueError(f"scalar must be {SCALAR_BYTES} bytes, got {len(scalar)}")
    out = ctypes.create_string_buffer(GROUP_ELEMENT_BYTES)
    if _load().crypto_scalarmult_ristretto255_base(out, scalar) != 0:
        raise ValueError("ristretto255 base scalar multiplication failed")
    return out.raw[:GROUP_ELEMENT_BYTES]


def is_valid_point(point: bytes) -> bool:
    """Whether ``point`` is a canonical encoding of a group element."""
    if len(point) != GROUP_ELEMENT_BYTES:
        return False
    return _load().crypto_core_ristretto255_is_valid_point(point) == 1
