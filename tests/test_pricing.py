"""Cost calculation from the price table."""

from decimal import Decimal
from pathlib import Path

import pytest

from memagent.llm.types import Usage
from memagent.pricing import UnknownModelPrice, cost_usd, load_price_table
from tests.conftest import REPO_ROOT, TEST_PRICES


def test_cached_tokens_are_billed_at_cached_rate_not_twice() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=1_000_000)
    assert cost_usd(TEST_PRICES, "fake", usage) == Decimal("0.10")


def test_full_formula_with_search_fee() -> None:
    usage = Usage(
        input_tokens=2_000_000, output_tokens=500_000, cached_input_tokens=500_000, search_calls=2
    )
    # 1.5M * $1 + 0.5M * $0.10 + 0.5M * $4 + 2 * $0.01
    assert cost_usd(TEST_PRICES, "fake", usage) == Decimal("3.57")


def test_unknown_model_raises() -> None:
    with pytest.raises(UnknownModelPrice):
        cost_usd(TEST_PRICES, "not-in-table", Usage(1, 1))


def test_load_price_table(tmp_path: Path) -> None:
    path = tmp_path / "prices.toml"
    path.write_text(
        '[models."m"]\ninput_per_mtok = 0.4\ncached_input_per_mtok = 0.1\noutput_per_mtok = 1.6\n'
        "[search]\nper_call_usd = 0.01\n"
    )
    table = load_price_table(path)
    assert table.models["m"].input_per_mtok == Decimal("0.4")  # exact, no float noise
    assert table.search_per_call_usd == Decimal("0.01")


def test_committed_price_table_parses() -> None:
    load_price_table(REPO_ROOT / "config" / "prices.toml")
