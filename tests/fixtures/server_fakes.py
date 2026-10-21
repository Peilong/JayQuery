"""Fakes for testing fl_server/server_app.py.

The server talks to the federation only through `Grid.get_node_ids()`,
`Grid.send_and_receive()` and `Grid.push_messages()`. FakeGrid implements that
three-method surface and records everything, so a whole VFL round loop can run
in-process with no SuperLink, no SuperNodes and no network.

Relies on the flwr stand-ins installed by tests/fixtures/conftest.py.
"""

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, RecordDict

SILO_A_POS = 0
SILO_B_POS = 1


def emb_reply(
    *,
    pos: int,
    src_node_id: int,
    n_rows: int = 8,
    embedding_dim: int = 4,
    labels: torch.Tensor | None = None,
    n_total: int | None = None,
    n_pos: int | None = None,
    n_neg: int | None = None,
    privacy: dict | None = None,
    budget_exhausted: bool = False,
    seed: int = 0,
    error: object | None = None,
) -> Message:
    """Build one silo's reply to a `query.generate_embeddings` message.

    Mirrors what fl_client.common.generate_embeddings actually sends: a
    float32 embedding, plus (Silo A only) the labels and class counts.

    Args:
        pos: Partition id — SILO_A_POS also attaches labels.
        labels: Explicit label vector; random 0/1 when omitted.
        budget_exhausted: Set the flag that makes the server stop the run.
        error: Non-None makes has_content() False (a dropped reply).
    """
    gen = torch.Generator().manual_seed(seed + pos)
    emb = torch.randn(n_rows, embedding_dim, generator=gen, dtype=torch.float32)

    content = {
        "arrays": ArrayRecord({"embedding": Array(emb.numpy())}),
    }

    cfg: dict = {"pos": pos}
    if budget_exhausted:
        cfg["budget-exhausted"] = True
    if n_total is not None:
        cfg["n-total"] = n_total

    if pos == SILO_A_POS:
        if labels is None:
            labels = (
                torch.rand(n_rows, generator=gen) < 0.3
            ).to(torch.float32)
        content["arrays"] = ArrayRecord(
            {
                "embedding": Array(emb.numpy()),
                "labels": Array(labels.numpy().astype(np.float32)),
            }
        )
        if n_pos is not None or n_neg is not None:
            cfg["n-pos"] = int(n_pos if n_pos is not None else labels.sum())
            cfg["n-neg"] = int(n_neg if n_neg is not None else (1 - labels).sum())

    content["config"] = ConfigRecord(cfg)
    if privacy is not None:
        content["privacy"] = ConfigRecord(dict(privacy))

    return Message(
        content=RecordDict(content),
        src_node_id=src_node_id,
        error=error,
    )


class FakeGrid:
    """Scriptable stand-in for flwr.serverapp.Grid.

    Args:
        node_ids: Node ids the server will see from get_node_ids().
        responder: Callable (messages, call_index) -> list[Message] used to
            answer send_and_receive. Defaults to `default_responder`, which
            replies to every query with a well-formed embedding reply.
    """

    def __init__(self, node_ids=(11, 22), responder=None):
        self._node_ids = list(node_ids)
        self.responder = responder or self.default_responder
        self.pushed: list[Message] = []
        self.sent: list[list[Message]] = []
        self.send_and_receive_calls = 0
        self.push_calls = 0
        # Per-node bottom-model stand-ins, so repeated rounds can return
        # embeddings that actually change in response to gradients.
        self.n_rows = 8
        self.embedding_dim = 4
        self.labels: torch.Tensor | None = None
        self.privacy_by_pos: dict[int, dict] = {}
        self.exhaust_after: int | None = None
        self.n_total = 40
        # Embeddings vary per call by default (each round is a different
        # Poisson batch). Pin this to get byte-identical replies every call.
        self.emb_seed: int | None = None

    # ── Grid surface ────────────────────────────────────────────────────────

    def get_node_ids(self):
        return list(self._node_ids)

    def send_and_receive(self, messages, timeout=None):
        self.send_and_receive_calls += 1
        msgs = list(messages)
        self.sent.append(msgs)
        return self.responder(msgs, self.send_and_receive_calls)

    def push_messages(self, messages):
        self.push_calls += 1
        self.pushed.extend(messages)

    # ── default behavior ────────────────────────────────────────────────────

    def default_responder(self, messages, call_index):
        """Answer every outgoing query with a valid per-silo reply."""
        exhausted = (
            self.exhaust_after is not None and call_index > self.exhaust_after
        )
        seed = call_index if self.emb_seed is None else self.emb_seed
        replies = []
        for i, msg in enumerate(messages):
            pos = SILO_A_POS if i == 0 else SILO_B_POS
            replies.append(
                emb_reply(
                    pos=pos,
                    src_node_id=msg.dst_node_id,
                    n_rows=self.n_rows,
                    embedding_dim=self.embedding_dim,
                    labels=self.labels if pos == SILO_A_POS else None,
                    n_total=self.n_total,
                    n_pos=None if self.labels is None else int(self.labels.sum()),
                    n_neg=None
                    if self.labels is None
                    else int((1 - self.labels).sum()),
                    privacy=self.privacy_by_pos.get(pos),
                    budget_exhausted=exhausted,
                    seed=seed,
                )
            )
        return replies

    # ── assertion helpers ───────────────────────────────────────────────────

    def queries_of_type(self, message_type: str) -> list[Message]:
        return [m for batch in self.sent for m in batch if m.message_type == message_type]

    def gradient_pushes(self) -> list[Message]:
        return [
            m
            for m in self.pushed
            if m.message_type == "train.apply_gradients" and "gradients" in m.content
        ]

    def stop_pushes(self) -> list[Message]:
        return [
            m
            for m in self.pushed
            if m.content["config"].get("stop-run", False)
        ]


def balanced_labels(n_rows: int = 8, n_pos: int = 3) -> torch.Tensor:
    """Deterministic label vector with a known positive count."""
    labels = torch.zeros(n_rows, dtype=torch.float32)
    labels[:n_pos] = 1.0
    return labels
