"""add gatekeeper columns to queue

This migration adds the four columns the message gatekeeper needs to track
per-queue-item classification state, plus the two indexes the polling loop
will use.

  status             — ready / pending / demoted / revived
  gate_verdict       — JSON payload produced by the Bonsai classifier
  gate_decided_at    — when the verdict was written
  reclassify_count   — how many times the gatekeeper has re-evaluated this row

The state machine:
  new message      → enqueued with status='pending' and no verdict
  gatekeeper runs  → status flips to 'ready' (A>B), 'pending' (|A-B|<δ), or 'demoted' (B>A)
  pending re-eval  → gatekeeper runs again; increments reclassify_count
  revived          → a demoted row is bumped back to processable by a later signal

Existing queue rows are backfilled to status='ready' so the pre-gatekeeper
worldview keeps working while the new plumbing is being stood up.

Revision ID: g7h8i9j0k1l2
Revises: h8i9j0k1l2m3
Create Date: 2026-04-19 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from migrations.utils import column_exists, get_schema, index_exists

revision: str = "g7h8i9j0k1l2"
down_revision: str | None = "h8i9j0k1l2m3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    # 1. Add columns — status nullable first so we can backfill without
    #    failing the NOT NULL constraint.
    if not column_exists("queue", "status", inspector):
        op.add_column(
            "queue",
            sa.Column("status", sa.TEXT(), nullable=True),
            schema=schema,
        )
    if not column_exists("queue", "gate_verdict", inspector):
        op.add_column(
            "queue",
            sa.Column(
                "gate_verdict",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
            schema=schema,
        )
    if not column_exists("queue", "gate_decided_at", inspector):
        op.add_column(
            "queue",
            sa.Column(
                "gate_decided_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            schema=schema,
        )
    if not column_exists("queue", "reclassify_count", inspector):
        op.add_column(
            "queue",
            sa.Column(
                "reclassify_count",
                sa.SmallInteger(),
                nullable=False,
                server_default=sa.text("0"),
            ),
            schema=schema,
        )

    # 2. Backfill pre-existing rows to 'ready' so they keep being processed
    #    under the old rules. The gatekeeper only influences NEW messages.
    op.execute(
        f"UPDATE \"{schema}\".queue SET status = 'ready' WHERE status IS NULL;"
    )

    # 3. Enforce NOT NULL + default 'pending' for new rows going forward.
    #    New rows will start as pending until the gatekeeper writes a verdict.
    op.alter_column(
        "queue",
        "status",
        existing_type=sa.TEXT(),
        nullable=False,
        server_default=sa.text("'pending'"),
        schema=schema,
    )

    # 4. Hot-path indexes.
    if not index_exists("queue", "ix_queue_pending_session", inspector):
        op.create_index(
            "ix_queue_pending_session",
            "queue",
            ["session_id", "id"],
            schema=schema,
            postgresql_where=sa.text(
                "status = 'pending' AND processed = false"
            ),
        )

    if not index_exists("queue", "ix_queue_demoted_decided_at", inspector):
        op.create_index(
            "ix_queue_demoted_decided_at",
            "queue",
            ["gate_decided_at"],
            schema=schema,
            postgresql_where=sa.text("status = 'demoted'"),
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if index_exists("queue", "ix_queue_demoted_decided_at", inspector):
        op.drop_index(
            "ix_queue_demoted_decided_at",
            table_name="queue",
            schema=schema,
        )
    if index_exists("queue", "ix_queue_pending_session", inspector):
        op.drop_index(
            "ix_queue_pending_session",
            table_name="queue",
            schema=schema,
        )

    if column_exists("queue", "reclassify_count", inspector):
        op.drop_column("queue", "reclassify_count", schema=schema)
    if column_exists("queue", "gate_decided_at", inspector):
        op.drop_column("queue", "gate_decided_at", schema=schema)
    if column_exists("queue", "gate_verdict", inspector):
        op.drop_column("queue", "gate_verdict", schema=schema)
    if column_exists("queue", "status", inspector):
        op.drop_column("queue", "status", schema=schema)
