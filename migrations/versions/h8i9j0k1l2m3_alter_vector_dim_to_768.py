"""alter embedding columns to 768-dim (hermes-stack customization)

Hermes-stack runs embeddings through Ollama-hosted `nomic-embed-text`, which
produces 768-dim vectors, not the 1536-dim `openai/text-embedding-3-small`
that plastic-labs/honcho defaults to in its upstream schema. This migration
ALTERs the two embedding columns down from Vector(1536) to Vector(768):

  * documents.embedding
  * message_embeddings.embedding

For a fresh install on this fork, alembic applies the upstream schema
(1536-dim empty columns) and then this migration flips them to 768. The
ALTER is a no-op if the columns are already 768-dim (i.e., running this
against a database that was bootstrapped via the earlier in-place-modified
migrations works without error).

IMPORTANT — existing data: if you are migrating an existing *vanilla*
upstream honcho database with 1536-dim rows to this fork, PostgreSQL cannot
shrink a populated pgvector column in place. Either
  (a) run `TRUNCATE documents, message_embeddings` before `alembic upgrade`,
      then re-derive observations from messages, or
  (b) drop and recreate the columns via a custom data migration.
Most hermes-stack users are on a fresh install, so (a) is the default path.

Revision ID: h8i9j0k1l2m3
Revises: e4eba9cfaa6f
Create Date: 2026-04-20 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "h8i9j0k1l2m3"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    """Shrink documents.embedding and message_embeddings.embedding to 768 dim."""
    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(1536),
        type_=Vector(768),
        schema=schema,
    )
    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(1536),
        type_=Vector(768),
        schema=schema,
    )


def downgrade() -> None:
    """Grow embedding columns back to 1536 dim. Will fail if columns contain
    768-dim data — downgrade is primarily for fresh-install correctness, not
    production rollback."""
    op.alter_column(
        "message_embeddings",
        "embedding",
        existing_type=Vector(768),
        type_=Vector(1536),
        schema=schema,
    )
    op.alter_column(
        "documents",
        "embedding",
        existing_type=Vector(768),
        type_=Vector(1536),
        schema=schema,
    )
