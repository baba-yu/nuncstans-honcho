"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_prompt(
    peer_id: str,
    target_assertions: str,
    context_messages: str,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    The prompt structurally separates the target peer's own utterances from
    other peers' messages. The LLM is instructed to extract facts ONLY from
    `target_assertions`; `context_messages` are provided solely for
    disambiguation (resolving pronouns, understanding what `peer_id` is
    responding to). This prevents the deriver from ingesting another peer's
    claims about `peer_id` as memory — critical when one of the other peers
    is an AI chat agent whose hallucinated "I remember you are X" lines
    would otherwise feed back into the fact store as new observations.

    Args:
        peer_id: The ID of the user being analyzed (the `observed` peer).
        target_assertions: Formatted messages authored by `peer_id`.
        context_messages: Formatted messages authored by any other peer in
            the session.

    Returns:
        Formatted prompt string for observation extraction.
    """
    return c(
        f"""
Extract **explicit atomic facts** about {peer_id} from their own assertions ONLY.

CRITICAL RULE — STRICT PEER FILTER:
- Only extract facts from the <target_assertions> block below. Those lines
  are what {peer_id} wrote.
- The <context_messages> block contains messages authored by OTHER peers
  in the same session (possibly an AI assistant or another user). Use them
  ONLY to resolve pronouns or understand what {peer_id} is responding to.
- You MUST NEVER extract a fact about {peer_id} from <context_messages>.
  Other peers may state things that are wrong, hallucinated, or about a
  different person. Only {peer_id}'s own assertions are evidence.
- If <target_assertions> is empty, return an empty observation list.

[EXPLICIT] DEFINITION: Facts about {peer_id} derivable directly from their own messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to {peer_id}. If {peer_id} is referencing
  someone or something else, make that clear.
- Observations must make sense on their own (future recall uses them verbatim).
- Contextualize each observation (e.g. "Ann is nervous about the job
  interview at the pharmacy" not just "Ann is nervous").

EXAMPLES:
- EXPLICIT: "I just had my 25th birthday last Saturday" → "{peer_id} is 25 years old", "{peer_id}'s birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "{peer_id} has a dog", "{peer_id} lives in NYC"
- EXPLICIT: "{peer_id} attended college" + general knowledge → "{peer_id} completed high school or equivalent"

<context_messages>
{context_messages}
</context_messages>

<target_assertions>
{target_assertions}
</target_assertions>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = minimal_deriver_prompt(
            peer_id="",
            target_assertions="",
            context_messages="",
        )
        return estimate_tokens(prompt)
    except Exception:
        return 300
