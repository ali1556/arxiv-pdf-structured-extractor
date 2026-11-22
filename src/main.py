from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict
import httpx
from apify import Actor

from src.chunker import generate_rag_chunks
from src.fetcher import (
    DEFAULT_USER_AGENT,
    download_pdf_bytes,
    fetch_arxiv_api_metadata,
    resolve_paper_url,
)
from src.models import ActorInput
from src.parser import parse_pdf_document

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arxiv_extractor")


async def main() -> None:
    async with Actor:
        raw_input = await Actor.get_input() or {}
        logger.info(f"Received raw Actor input: {raw_input}")

        # Validate input schema
        try:
            actor_input = ActorInput(**raw_input)
        except Exception as e:
            logger.error(f"Invalid input provided to Actor: {e}")
            await Actor.fail(status_message=f"Input validation error: {e}")
            return

        urls_or_ids = actor_input.urls_or_ids[: actor_input.max_papers]
        logger.info(f"Processing {len(urls_or_ids)} papers (max: {actor_input.max_papers})...")

        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/pdf,application/xml,text/html,*/*",
        }

        # Setup httpx proxy if provided via Apify proxy config
        proxy_url = None
        if actor_input.proxyConfiguration:
            try:
                proxy_config = await Actor.create_proxy_configuration(
                    actor_proxy_input=actor_input.proxyConfiguration
                )
                if proxy_config:
                    proxy_url = await proxy_config.new_url()
                    logger.info("Configured Apify Proxy for outgoing requests.")
            except Exception as proxy_err:
                logger.warning(f"Could not initialize Apify proxy: {proxy_err}. Proceeding direct.")

        success_count = 0
        failed_count = 0

        async with httpx.AsyncClient(
            headers=headers,
            proxy=proxy_url,
            follow_redirects=True,
            timeout=actor_input.timeout_secs,
        ) as client:
            for index, item in enumerate(urls_or_ids, start=1):
                clean_item = item.strip()
                if not clean_item:
                    continue

                start_time = time.time()
                logger.info(f"[{index}/{len(urls_or_ids)}] Resolving source: {clean_item}")

                try:
                    pdf_url, arxiv_id = resolve_paper_url(clean_item)
                    logger.info(f"Resolved PDF URL: {pdf_url} (ArXiv ID: {arxiv_id})")

                    # Step 1: If ArXiv ID is present, fetch authoritative metadata
                    api_meta: Dict[str, Any] | None = None
                    if arxiv_id:
                        api_meta = await fetch_arxiv_api_metadata(client, arxiv_id)
                        # Polite throttle between ArXiv API queries
                        await asyncio.sleep(1.0)

                    # Step 2: Download PDF binary stream
                    pdf_bytes = await download_pdf_bytes(
                        client=client,
                        pdf_url=pdf_url,
                        timeout_secs=actor_input.timeout_secs,
                    )

                    duration = time.time() - start_time

                    # Step 3: Fast deterministic PDF layout & content parsing
                    structured_output = parse_pdf_document(
                        pdf_bytes=pdf_bytes,
                        source_input=clean_item,
                        pdf_url=pdf_url,
                        arxiv_id=arxiv_id,
                        api_metadata=api_meta,
                        duration_secs=duration,
                    )

                    # Step 4: Optional RAG chunking
                    if actor_input.extract_chunks:
                        chunks = generate_rag_chunks(
                            sections=structured_output.sections,
                            chunk_size=actor_input.chunk_size,
                            chunk_overlap=actor_input.chunk_overlap,
                        )
                        structured_output.chunks = chunks

                    # Step 5: Push result to Apify Dataset
                    dataset_record = structured_output.model_dump(mode="json")
                    await Actor.push_data(dataset_record)
                    logger.info(
                        f"Successfully parsed '{structured_output.metadata.title[:50]}...' "
                        f"({structured_output.stats.pages} pages, {len(structured_output.citations)} citations)"
                    )

                    # Step 6: Monetization (Pay-Per-Event)
                    # ONLY charge event AFTER successful parse and dataset push!
                    try:
                        await Actor.charge(event_name="paper-parsed", count=1)
                        logger.info("Successfully charged PPE event 'paper-parsed' (count=1)")
                    except Exception as charge_err:
                        # Local runs or accounts without PPE will log notice without failing the run
                        logger.warning(f"Actor.charge notice (expected during local run): {charge_err}")

                    success_count += 1

                except Exception as err:
                    failed_count += 1
                    logger.error(f"Failed to process '{clean_item}': {err}", exc_info=True)
                    # Push error record to dataset for transparency (NO PPE CHARGE)
                    await Actor.push_data({
                        "status": "FAILED",
                        "source_input": clean_item,
                        "error_message": str(err),
                        "timestamp": time.time(),
                    })

                # Polite delay between consecutive paper downloads to avoid 429 rate limits
                if index < len(urls_or_ids):
                    await asyncio.sleep(1.5)

        logger.info(
            f"Actor run finished. Summary: {success_count} succeeded, {failed_count} failed."
        )


if __name__ == "__main__":
    asyncio.run(main())
