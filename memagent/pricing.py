"""Price table and cost calculation.

Prices live in a TOML file, not in code, because they change more often than the code
and because model names must never be hardcoded.
"""

import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from memagent.llm.types import Usage

ONE_MILLION = Decimal(1_000_000)


class UnknownModelPrice(KeyError):
    """No price entry for this model. Raised early so we never spend without tracking."""


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: Decimal
    cached_input_per_mtok: Decimal
    output_per_mtok: Decimal


@dataclass(frozen=True)
class PriceTable:
    models: dict[str, ModelPrice]
    search_per_call_usd: Decimal

    def require(self, model: str) -> ModelPrice:
        if model not in self.models:
            raise UnknownModelPrice(f"no price entry for model {model!r} in the price table")
        return self.models[model]


def load_price_table(path: Path) -> PriceTable:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    models = {
        name: ModelPrice(
            # str() first: Decimal(float) would carry binary rounding noise into costs.
            input_per_mtok=Decimal(str(entry["input_per_mtok"])),
            cached_input_per_mtok=Decimal(str(entry["cached_input_per_mtok"])),
            output_per_mtok=Decimal(str(entry["output_per_mtok"])),
        )
        for name, entry in raw.get("models", {}).items()
    }
    search_fee = Decimal(str(raw["search"]["per_call_usd"]))
    return PriceTable(models=models, search_per_call_usd=search_fee)


def cost_usd(table: PriceTable, model: str, usage: Usage) -> Decimal:
    """USD cost of one call.

    OpenAI reports cached tokens as a subset of input tokens, so the uncached part is
    `input - cached`; billing the full input at the normal rate would double-count.
    """
    price = table.require(model)
    uncached = usage.input_tokens - usage.cached_input_tokens
    token_cost = (
        uncached * price.input_per_mtok
        + usage.cached_input_tokens * price.cached_input_per_mtok
        + usage.output_tokens * price.output_per_mtok
    ) / ONE_MILLION
    return token_cost + usage.search_calls * table.search_per_call_usd
