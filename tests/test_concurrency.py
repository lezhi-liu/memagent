"""Two requests for the same session at once must not interleave or collide on seq."""

import asyncio

from memagent.llm.fake import FakeLLM, reply
from tests.conftest import MakeClient, MakeUser


async def test_same_session_turns_are_serialized(
    make_user: MakeUser, make_client: MakeClient
) -> None:
    _, key = await make_user()
    # The delay keeps each turn "inside the model call" long enough that, without the
    # session lock, the second turn would write its user_message in the middle of the first.
    llm = FakeLLM(lambda _messages: reply("answer"), delay=0.2)
    client = await make_client(llm, key)
    session_id = (await client.post("/sessions")).json()["id"]

    responses = await asyncio.gather(
        client.post(f"/sessions/{session_id}/messages", json={"text": "first"}),
        client.post(f"/sessions/{session_id}/messages", json={"text": "second"}),
    )
    assert [r.status_code for r in responses] == [200, 200]

    events = (await client.get(f"/sessions/{session_id}/events")).json()
    assert [e["seq"] for e in events] == [1, 2, 3, 4]  # contiguous, no duplicates
    # Each turn is contiguous: user, assistant, user, assistant.
    assert [e["type"] for e in events] == [
        "user_message",
        "assistant_message",
        "user_message",
        "assistant_message",
    ]
    # The second turn's model call saw the whole first turn.
    assert len(llm.calls[1]) == len(llm.calls[0]) + 2
