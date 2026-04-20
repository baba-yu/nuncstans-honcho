import logging
import time

from sqlalchemy import select

from src import crud
from src.config import ConfiguredModelSettings, settings
from src.crud.representation import RepresentationManager
from src.dependencies import tracked_db
from src.llm import honcho_llm_call
from src.models import Message, QueueItem
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import RepresentationCompletedEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.telemetry.sentry import with_sentry_transaction
from src.utils.agent_tools import TOOLS, create_tool_executor
from src.utils.config_helpers import get_configuration
from src.utils.formatting import format_new_turn_with_timestamp
from src.utils.representation import PromptRepresentation, Representation
from src.utils.tokens import track_deriver_input_tokens

from .prompts import estimate_minimal_deriver_prompt_tokens, minimal_deriver_prompt

logger = logging.getLogger(__name__)


def _get_deriver_model_config() -> ConfiguredModelSettings:
    return settings.DERIVER.MODEL_CONFIG


@with_sentry_transaction("minimal_deriver_batch", op="deriver")
async def process_representation_tasks_batch(
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_item_message_ids: list[int],
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple collections.

    Args:
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer peer IDs (collections to save to).
        observed: The observed peer ID.
        queue_item_message_ids: Message IDs from queue items being processed
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

    # Get configuration if not provided
    # TODO: this appears to be a very rare edge case coming out of `get_queue_item_batch` in queue_manager.py,
    # possible that we can remove this and require configuration to come through with the payload.
    if message_level_configuration is None:
        async with tracked_db("minimal_deriver.get_config") as db:
            message_level_configuration = get_configuration(
                None,
                await crud.get_session(
                    db, latest_message.session_name, latest_message.workspace_name
                ),
                await crud.get_workspace(
                    db, workspace_name=latest_message.workspace_name
                ),
            )

    # Skip if disabled
    if message_level_configuration.reasoning.enabled is False:
        return

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "starting_message_id",
        earliest_message.id,
        "id",
    )
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "ending_message_id",
        latest_message.id,
        "id",
    )

    # Split messages by authorship: the `observed` peer's own messages are the
    # only evidence source; everyone else (including AI chat agents whose
    # hallucinated "I remember you are X" lines would otherwise seed false
    # memories) goes in a separate context block the prompt explicitly tells
    # the LLM not to extract from.
    target_msgs = [m for m in messages if m.peer_name == observed]
    context_msgs = [m for m in messages if m.peer_name != observed]

    if not target_msgs:
        logger.info(
            "Deriver batch for %s contains no messages authored by the observed peer; "
            "skipping extraction (messages %s:%s in %s/%s)",
            observed,
            earliest_message.id,
            latest_message.id,
            latest_message.workspace_name,
            latest_message.session_name,
        )
        return

    target_formatted = "\n".join(
        format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
        for msg in target_msgs
    )
    context_formatted = (
        "\n".join(
            format_new_turn_with_timestamp(msg.content, msg.created_at, msg.peer_name)
            for msg in context_msgs
        )
        or "(no other-peer messages in this window)"
    )

    # Track token usage - count only tokens from messages being processed
    prompt_tokens = estimate_minimal_deriver_prompt_tokens()
    queue_item_message_ids_set = set(queue_item_message_ids)
    messages_tokens = sum(
        msg.token_count for msg in messages if msg.id in queue_item_message_ids_set
    )
    track_deriver_input_tokens(
        task_type=DeriverTaskTypes.INGESTION,
        components={
            DeriverComponents.PROMPT: prompt_tokens,
            DeriverComponents.MESSAGES: messages_tokens,
        },
    )

    # Build prompt
    prompt = minimal_deriver_prompt(
        peer_id=observed,
        target_assertions=target_formatted,
        context_messages=context_formatted,
    )

    context_prep_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "context_preparation",
        context_prep_duration,
        "ms",
    )

    # validation on settings means max_tokens will always be > 0
    base_model_config = _get_deriver_model_config()
    max_tokens = base_model_config.max_output_tokens or settings.LLM.DEFAULT_MAX_TOKENS
    model_config = base_model_config.model_copy(
        update={
            "stop_sequences": ["   \n", "\n\n\n\n"],
        }
    )

    # Single LLM call
    llm_start = time.perf_counter()
    response = await honcho_llm_call(
        model_config=model_config,
        prompt=prompt,
        max_tokens=max_tokens,
        track_name="Minimal Deriver",
        response_model=PromptRepresentation,
        json_mode=True,
        max_input_tokens=settings.DERIVER.MAX_INPUT_TOKENS,
        enable_retry=True,
        retry_attempts=3,
        trace_name="minimal_deriver",
    )
    llm_duration = (time.perf_counter() - llm_start) * 1000

    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "llm_call_duration",
        llm_duration,
        "ms",
    )

    # Prometheus metrics
    if settings.METRICS.ENABLED:
        prometheus_metrics.record_deriver_tokens(
            count=response.output_tokens,
            task_type=DeriverTaskTypes.INGESTION.value,
            token_type=TokenTypes.OUTPUT.value,
            component=DeriverComponents.OUTPUT_TOTAL.value,
        )

    message_ids = [m.id for m in messages if m.peer_name == observed]

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        message_ids,
        latest_message.session_name,
        latest_message.created_at,
    )

    if observations.is_empty() or not message_ids:
        logger.warning(
            "Deriver generated zero observations for messages %s:%s in %s/%s!",
            earliest_message.id,
            latest_message.id,
            latest_message.workspace_name,
            latest_message.session_name,
        )
    else:
        # Save to all observer collections
        for observer in observers:
            representation_manager = RepresentationManager(
                workspace_name=latest_message.workspace_name,
                observer=observer,
                observed=observed,
            )

            try:
                await representation_manager.save_representation(
                    observations,
                    message_ids,
                    latest_message.session_name,
                    latest_message.created_at,
                    message_level_configuration,
                )
            except Exception as e:
                logger.error(
                    "Failed to save representation for observer %s: %s", observer, e
                )

    # Log metrics
    overall_duration = (time.perf_counter() - overall_start) * 1000
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "total_processing_time",
        overall_duration,
        "ms",
    )

    total_observations = len(observations.explicit) + len(observations.deductive)
    accumulate_metric(
        f"minimal_deriver_{latest_message.id}_{observed}",
        "observation_count",
        total_observations,
        "count",
    )

    if settings.DERIVER.LOG_OBSERVATIONS:
        # Log messages fed into deriver
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "messages",
            f"TARGET:\n{target_formatted}\n\nCONTEXT:\n{context_formatted}",
            "blob",
        )
        # Log actual observations created as blob metrics
        accumulate_metric(
            f"minimal_deriver_{latest_message.id}_{observed}",
            "explicit_observations",
            "\n".join(f"  • {obs}" for obs in observations.explicit),
            "blob",
        )

    log_performance_metrics("minimal_deriver", f"{latest_message.id}_{observed}")

    # Emit telemetry event
    emit(
        RepresentationCompletedEvent(
            workspace_name=latest_message.workspace_name,
            session_name=latest_message.session_name,
            observed=observed,
            queue_items_processed=len(queue_item_message_ids),
            earliest_message_id=earliest_message.public_id,
            latest_message_id=latest_message.public_id,
            message_count=len(messages),
            explicit_conclusion_count=len(observations.explicit),
            context_preparation_ms=context_prep_duration,
            llm_call_ms=llm_duration,
            total_duration_ms=overall_duration,
            input_tokens=messages_tokens,
            output_tokens=response.output_tokens,
        )
    )

    # Gatekeeper-driven supersede pass. If any queue row flagged its message as
    # correction_of_prior, fire a secondary, tool-only LLM call so Bonsai can
    # soft-delete the now-contradicted observations. We run AFTER the main save
    # so the new observation already exists alongside the old ones at the moment
    # the model chooses which IDs to retract.
    if queue_item_message_ids:
        correction_mids = await _fetch_correction_message_ids(queue_item_message_ids)
        if correction_mids:
            correction_messages = [m for m in messages if m.id in correction_mids]
            if correction_messages:
                try:
                    await _run_supersede_pass(
                        correction_messages=correction_messages,
                        workspace_name=latest_message.workspace_name,
                        session_name=latest_message.session_name,
                        observers=observers,
                        observed=observed,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Supersede pass failed for message(s) %s: %s",
                        sorted(correction_mids),
                        e,
                    )


async def _fetch_correction_message_ids(
    queue_item_message_ids: list[int],
) -> set[int]:
    """Return the subset of queue_item_message_ids whose gate_verdict marks
    them as correction_of_prior=true."""
    async with tracked_db("deriver.check_corrections") as db:
        result = await db.execute(
            select(QueueItem.message_id, QueueItem.gate_verdict).where(
                QueueItem.message_id.in_(queue_item_message_ids)
            )
        )
        return {
            mid
            for mid, verdict in result.all()
            if mid is not None
            and isinstance(verdict, dict)
            and verdict.get("correction_of_prior") is True
        }


async def _run_supersede_pass(
    correction_messages: list[Message],
    workspace_name: str,
    session_name: str,
    observers: list[str],
    observed: str,
) -> None:
    """Secondary LLM pass that lets the deriver retract observations contradicted
    by an explicit user correction.

    Runs once per observer (each observer holds an independent copy of the
    peer's observations). Restricted to the ``supersede_observations`` tool to
    keep this pass strictly about retraction — no new observations, no peer
    card edits.
    """
    correction_text = "\n".join(m.content for m in correction_messages)
    supersede_tools = [TOOLS["supersede_observations"]]
    deriver_model_config = _get_deriver_model_config()

    for observer in observers:
        async with tracked_db("deriver.supersede.fetch_recent") as db:
            documents = await crud.query_documents_recent(
                db=db,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                limit=30,
            )
            # Materialize id + content while the session is open; Document
            # instances become detached once the `async with` block exits.
            obs_rows: list[tuple[str, str]] = [(d.id, d.content) for d in documents]
        if not obs_rows:
            continue

        obs_block = "\n".join(
            f"- {doc_id} :: {content}" for doc_id, content in obs_rows
        )

        prompt = (
            "You keep a peer's memory accurate.\n\n"
            f"The peer ({observed}) just posted a CORRECTION:\n"
            f'"""\n{correction_text}\n"""\n\n'
            "Here are their existing observations. Each line is formatted as "
            "`<observation_id> :: <content>`. The observation_id is the exact "
            "string before the ` :: ` separator — pass it verbatim to the tool, "
            "with no prefix or brackets.\n\n"
            f"{obs_block}\n\n"
            "Identify observations that are now directly CONTRADICTED by the "
            "correction (e.g. a prior name/location/profession claim that the "
            "correction explicitly revises). Call supersede_observations ONCE "
            "with a list of the raw observation_ids to retract and a short "
            "reason. Do NOT supersede observations that are merely topically "
            "related but not contradicted. If nothing needs retracting, reply "
            'with "no supersede needed" and do not call any tool.'
        )

        tool_executor = await create_tool_executor(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            agent_type="deriver",
            parent_category="deriver.supersede",
        )

        await honcho_llm_call(
            model_config=deriver_model_config,
            prompt=prompt,
            max_tokens=512,
            tools=supersede_tools,
            tool_choice="auto",
            tool_executor=tool_executor,
            max_tool_iterations=2,
            track_name="Deriver Supersede Pass",
            temperature=deriver_model_config.temperature,
            enable_retry=True,
            retry_attempts=2,
            trace_name="deriver_supersede",
        )
