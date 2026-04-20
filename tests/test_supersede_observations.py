"""Tests for the ``supersede_observations`` agent tool.

Scenarios covered:

1. Normal supersede: an existing, non-deleted document is soft-deleted and its
   ``internal_metadata`` is merged with supersede linkage fields.
2. Unknown document id: the id is reported under ``not_found`` and no DB rows
   are touched.
3. Already-deleted document: the id is reported under ``already_deleted`` so
   callers can distinguish it from the ``not_found`` case.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import Callable
from typing import Any

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.utils.agent_tools import (
    DERIVER_TOOLS,
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    DREAMER_TOOLS,
    TOOLS,
    ToolContext,
    _handle_supersede_observations,  # pyright: ignore[reportPrivateUsage]
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def supersede_test_data(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> Any:
    """Create workspace/peers/collection and a single observation to supersede."""
    workspace, peer1 = sample_data

    # Create the observed peer.
    peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(peer2)
    await db_session.flush()

    # Create collection (peer1 observes peer2).
    collection = models.Collection(
        workspace_name=workspace.name,
        observer=peer1.name,
        observed=peer2.name,
    )
    db_session.add(collection)
    await db_session.flush()

    # Create one explicit observation that we will try to supersede.
    doc = models.Document(
        workspace_name=workspace.name,
        observer=peer1.name,
        observed=peer2.name,
        content="User lives in Kyoto",
        embedding=[0.1] * 1536,
        level="explicit",
        internal_metadata={
            "message_ids": [],
            "message_created_at": "2025-01-01T00:00:00+00:00",
        },
    )
    db_session.add(doc)
    await db_session.flush()
    await db_session.refresh(doc)

    # Commit so the tool handler's independent tracked_db session can see the row.
    await db_session.commit()

    yield workspace, peer1, peer2, doc


@pytest.fixture
def make_ctx(supersede_test_data: Any) -> Callable[..., ToolContext]:
    """Factory producing a ``ToolContext`` bound to the supersede test data."""
    workspace, peer1, peer2, _ = supersede_test_data
    shared_lock = asyncio.Lock()

    def _factory() -> ToolContext:
        return ToolContext(
            workspace_name=workspace.name,
            observer=peer1.name,
            observed=peer2.name,
            session_name=None,
            current_messages=None,
            include_observation_ids=True,
            history_token_limit=8192,
            db_lock=shared_lock,
        )

    return _factory


# =============================================================================
# Tool exposure (structural) tests — verify supersede is deriver-only
# =============================================================================


class TestSupersedeObservationsExposure:
    """Structural tests guaranteeing the deriver-only exposure contract."""

    def test_tool_schema_is_registered(self) -> None:
        """The tool definition is present in the shared TOOLS registry."""
        assert "supersede_observations" in TOOLS
        schema = TOOLS["supersede_observations"]
        assert schema["name"] == "supersede_observations"
        required = schema["input_schema"]["required"]
        assert "document_ids_to_supersede" in required
        assert "reason" in required

    def test_deriver_tools_includes_supersede(self) -> None:
        """DERIVER_TOOLS contains supersede_observations."""
        names = [t["name"] for t in DERIVER_TOOLS]
        assert "supersede_observations" in names

    def test_dialectic_tools_excludes_supersede(self) -> None:
        """Dialectic agents must not see supersede_observations."""
        for tools in (DIALECTIC_TOOLS, DIALECTIC_TOOLS_MINIMAL):
            names = [t["name"] for t in tools]
            assert "supersede_observations" not in names

    def test_dreamer_tools_excludes_supersede(self) -> None:
        """Dreamer agent must not see supersede_observations."""
        names = [t["name"] for t in DREAMER_TOOLS]
        assert "supersede_observations" not in names


# =============================================================================
# Behavioural tests for _handle_supersede_observations
# =============================================================================


@pytest.mark.asyncio
class TestSupersedeObservations:
    """Unit tests for the ``_handle_supersede_observations`` handler."""

    async def test_supersedes_existing_observation(
        self,
        db_session: AsyncSession,
        supersede_test_data: Any,
        make_ctx: Callable[..., ToolContext],
    ) -> None:
        """A live observation is soft-deleted and metadata records the supersede."""
        _, _, _, doc = supersede_test_data
        doc_id = doc.id
        ctx = make_ctx()

        result = await _handle_supersede_observations(
            ctx,
            {
                "document_ids_to_supersede": [doc_id],
                "reason": "user corrected location to Osaka",
            },
        )

        payload = json.loads(result)
        assert payload["superseded"] == [doc_id]
        assert payload["not_found"] == []
        assert payload["already_deleted"] == []

        # Re-read the row from the DB to verify soft-delete + metadata merge.
        # Rollback + expire-all ensures we do not read from the session's
        # identity map — the tool handler committed through its own session.
        await db_session.rollback()
        db_session.expire_all()
        stmt = select(models.Document).where(models.Document.id == doc_id)
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.deleted_at is not None
        assert row.internal_metadata.get("supersede_reason") == (
            "user corrected location to Osaka"
        )
        assert "superseded_by" in row.internal_metadata
        assert row.internal_metadata.get("deleted_reason") == "superseded"
        # Original metadata keys are preserved (JSONB merge is shallow).
        assert row.internal_metadata.get("message_created_at") == (
            "2025-01-01T00:00:00+00:00"
        )

    async def test_unknown_id_goes_to_not_found(
        self,
        supersede_test_data: Any,
        make_ctx: Callable[..., ToolContext],
    ) -> None:
        """An id that does not exist in the workspace reports as ``not_found``."""
        _ = supersede_test_data
        ctx = make_ctx()

        # 21-char nanoid-shaped but unused id.
        missing_id = "a" * 21

        result = await _handle_supersede_observations(
            ctx,
            {
                "document_ids_to_supersede": [missing_id],
                "reason": "probing a missing id",
            },
        )

        payload = json.loads(result)
        assert payload["superseded"] == []
        assert payload["not_found"] == [missing_id]
        assert payload["already_deleted"] == []

    async def test_already_deleted_id_is_reported_separately(
        self,
        db_session: AsyncSession,
        supersede_test_data: Any,
        make_ctx: Callable[..., ToolContext],
    ) -> None:
        """An already soft-deleted document is a no-op and reported distinctly."""
        _, _, _, doc = supersede_test_data
        doc_id = doc.id

        # Pre-mark the document as deleted.
        doc.deleted_at = datetime.datetime.now(datetime.timezone.utc)
        await db_session.commit()

        ctx = make_ctx()

        result = await _handle_supersede_observations(
            ctx,
            {
                "document_ids_to_supersede": [doc_id],
                "reason": "this should be a no-op",
            },
        )

        payload = json.loads(result)
        assert payload["superseded"] == []
        assert payload["not_found"] == []
        assert payload["already_deleted"] == [doc_id]

        # Ensure the original deleted_at was not overwritten and no supersede
        # metadata was added.
        await db_session.rollback()
        db_session.expire_all()
        stmt = select(models.Document).where(models.Document.id == doc_id)
        row = (await db_session.execute(stmt)).scalar_one()
        assert row.deleted_at is not None
        assert "supersede_reason" not in row.internal_metadata

    async def test_empty_id_list_returns_error(
        self, make_ctx: Callable[..., ToolContext]
    ) -> None:
        """An empty id list is rejected without touching the DB."""
        ctx = make_ctx()

        result = await _handle_supersede_observations(
            ctx,
            {"document_ids_to_supersede": [], "reason": "doesn't matter"},
        )

        assert result.startswith("ERROR")

    async def test_missing_reason_returns_error(
        self, supersede_test_data: Any, make_ctx: Callable[..., ToolContext]
    ) -> None:
        """A missing / whitespace-only reason is rejected."""
        _, _, _, doc = supersede_test_data
        ctx = make_ctx()

        result = await _handle_supersede_observations(
            ctx,
            {"document_ids_to_supersede": [doc.id], "reason": "   "},
        )

        assert result.startswith("ERROR")
