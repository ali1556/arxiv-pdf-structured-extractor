import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from apify import Actor
from src.main import main


@pytest.fixture(autouse=True)
def disable_apify_process_exit(monkeypatch):
    """Prevent Apify SDK from calling sys.exit() during unit tests."""
    monkeypatch.setattr(Actor, "_exit_process", False)


def test_pay_per_event_schema_exists():
    """Verify pay_per_event.json contains correct event definition and price."""
    ppe_path = Path(__file__).resolve().parent.parent / ".actor" / "pay_per_event.json"
    assert ppe_path.exists(), "pay_per_event.json must exist in .actor/"

    with open(ppe_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("actorSpecification") == 1
    events = data.get("events", {})
    assert "paper-parsed" in events, "Event 'paper-parsed' must be configured in pay_per_event.json"
    assert events["paper-parsed"]["priceUsd"] > 0
    assert "eventTitle" in events["paper-parsed"]


@pytest.mark.asyncio
async def test_successful_paper_triggers_ppe_charge():
    """Verifies that Actor.charge('paper-parsed', count=1) is called on success."""
    fake_pdf = b"%PDF-1.4 Fake academic paper stream"

    with patch("src.main.Actor.get_input", new_callable=AsyncMock) as mock_get_input, \
         patch("src.main.Actor.push_data", new_callable=AsyncMock) as mock_push_data, \
         patch("src.main.Actor.charge", new_callable=AsyncMock) as mock_charge, \
         patch("src.main.download_pdf_bytes", new_callable=AsyncMock) as mock_download, \
         patch("src.main.fetch_arxiv_api_metadata", new_callable=AsyncMock) as mock_meta, \
         patch("src.main.parse_pdf_document") as mock_parse:

        mock_get_input.return_value = {
            "urls_or_ids": ["https://arxiv.org/abs/2312.12345"],
            "extract_chunks": False,
        }
        mock_download.return_value = fake_pdf
        mock_meta.return_value = {"title": "Test Title", "abstract": "Test Abstract"}

        # Mock structured output
        mock_output = AsyncMock()
        mock_output.model_dump.return_value = {"status": "SUCCESS", "title": "Test Title"}
        mock_output.metadata.title = "Test Title"
        mock_output.stats.pages = 10
        mock_output.citations = []
        mock_parse.return_value = mock_output

        await main()

        # Must push data first, then charge
        assert mock_push_data.call_count == 1
        assert mock_charge.call_count == 1
        mock_charge.assert_called_once_with(event_name="paper-parsed", count=1)


@pytest.mark.asyncio
async def test_failed_paper_never_charges_ppe():
    """Verifies that Actor.charge is NEVER called when a download or parsing error occurs."""
    with patch("src.main.Actor.get_input", new_callable=AsyncMock) as mock_get_input, \
         patch("src.main.Actor.push_data", new_callable=AsyncMock) as mock_push_data, \
         patch("src.main.Actor.charge", new_callable=AsyncMock) as mock_charge, \
         patch("src.main.download_pdf_bytes", new_callable=AsyncMock) as mock_download:

        mock_get_input.return_value = {
            "urls_or_ids": ["https://corrupted-or-404-link.com/bad.pdf"],
            "extract_chunks": False,
        }
        mock_download.side_effect = RuntimeError("Failed 404 Not Found")

        await main()

        # Dataset should record failure status
        assert mock_push_data.call_count == 1
        pushed_data = mock_push_data.call_args[0][0]
        assert pushed_data.get("status") == "FAILED"

        # PPE Charge MUST NOT be called!
        assert mock_charge.call_count == 0
