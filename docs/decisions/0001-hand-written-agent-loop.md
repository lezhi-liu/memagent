# 0001: A hand-written agent loop instead of a framework

- Status: accepted (S1)
- Code: `memagent/agent/loop.py`, `memagent/tools/base.py`

## Context

The agent needs a loop that calls the model, runs the tools it asks for, and feeds the results back. Later slices hook into exactly this loop: memory recall before the call, memory writes after it, evals around it, and cost tracking on every call. This project is also meant to be explained line by line in an interview.

## Options

1. **A framework** (LangChain/LangGraph, the OpenAI Agents SDK, LlamaIndex agents). Fast to start, with many integrations.
2. **A provider's built-in loop** (an SDK "tool runner" that runs tools automatically).
3. **Write the loop by hand**: about 100 lines in `loop.py`, plus a small `ToolRegistry`.

## Choice

Option 3.

- **Every step has to be persisted**, in order, through the scrubbing gateway, under a per-session lock. Frameworks keep their own message state; we would have to fight it so that history is rebuilt from our events table on every call (decision 0003).
- **Every model call must write an `llm_calls` row** with our module labels and our price table. Hooking a framework's callbacks would hide the most important line of the system.
- **Failure semantics are ours.** Tool exceptions become `is_error` results, the step limit writes an `error` event, and an unpaired tool call is dropped from history. These are a few explicit lines here; in a framework they are configuration we would have to discover.
- **Provider independence.** The loop only sees `memagent.llm.types`. Switching providers means one new file.
- **Explainability.** The whole control flow fits on one screen.

## Cost

- We write and maintain things a framework would give us for free: retries around tools, parallel tool execution, streaming. S1 runs tools sequentially and does not stream.
- No ready-made integrations. Each tool is ours to write and test.
- Ideas from frameworks (handoffs, guardrails) have to be re-implemented if we want them.
