"""Command-line interface.

- `memagent user create <name>` is an admin command: it talks to the database directly,
  because user creation has no API endpoint in S1.
- `memagent chat` is an ordinary API client: it goes through HTTP and the bearer key,
  exactly like any other client would.
"""

import asyncio
from decimal import Decimal
from typing import Annotated, Any

import httpx
import typer

from memagent import store
from memagent.auth import generate_api_key, hash_api_key
from memagent.config import Settings
from memagent.db.engine import make_engine, make_sessionmaker

app = typer.Typer(help="memagent: a multi-user agent with long-term memory.", no_args_is_help=True)
user_app = typer.Typer(help="Manage users.", no_args_is_help=True)
app.add_typer(user_app, name="user")

DEFAULT_API_URL = "http://127.0.0.1:8000"
TURN_TIMEOUT_SECONDS = 300.0


async def _create_user(name: str) -> tuple[str, str]:
    engine = make_engine(Settings().database_url)
    try:
        api_key = generate_api_key()
        async with make_sessionmaker(engine)() as db:
            user = await store.create_user(db, name, hash_api_key(api_key))
        return str(user.id), api_key
    finally:
        await engine.dispose()


@user_app.command("create")
def user_create(name: str) -> None:
    """Create a user and print its API key. The key is shown once and never stored."""
    user_id, api_key = asyncio.run(_create_user(name))
    typer.echo(f"user id: {user_id}")
    typer.echo(f"api key: {api_key}")
    typer.echo("Save the key now; only its hash is stored, so it cannot be shown again.")


def _print_turn(turn: dict[str, Any]) -> None:
    for step in turn["steps"]:
        marker = "error" if step["is_error"] else "ok"
        typer.secho(f"  [tool] {step['tool']} ({marker})", fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"agent> {turn['reply']}")
    cost = Decimal(str(turn["cost_usd"]))
    typer.secho(f"  [cost] ${cost:.6f} this turn", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def chat(
    api_key: Annotated[str, typer.Option(envvar="MEMAGENT_API_KEY", help="Your API key")],
    api_url: Annotated[str, typer.Option(envvar="MEMAGENT_API_URL")] = DEFAULT_API_URL,
) -> None:
    """Start a new session and chat. Type /exit (or Ctrl-D) to quit."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=api_url, headers=headers, timeout=TURN_TIMEOUT_SECONDS) as client:
        response = client.post("/sessions")
        if response.status_code == 401:
            raise typer.Exit(_fail("invalid API key"))
        response.raise_for_status()
        session_id = response.json()["id"]
        typer.secho(f"session {session_id} (new session: no memory of earlier ones)", dim=True)

        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo()
                break
            if text in ("/exit", "/quit"):
                break
            if not text:
                continue
            response = client.post(f"/sessions/{session_id}/messages", json={"text": text})
            if response.status_code != 200:
                _fail(f"HTTP {response.status_code}: {response.json().get('detail', '')}")
                continue
            _print_turn(response.json())


def _fail(message: str) -> int:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return 1


if __name__ == "__main__":
    app()
