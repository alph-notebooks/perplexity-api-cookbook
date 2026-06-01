#!/usr/bin/env python3
"""
Finance Chart (Sandbox) - Plot a stock's closing-price history using the
Perplexity Agent API ``sandbox`` tool, then render the chart locally.

The agent loop runs entirely inside one **background** Agent API request:

  1. The model is given the ``sandbox`` tool — a full agentic Python
     environment that includes the Perplexity SDK (web search + URL fetch).
  2. Inside the sandbox it searches for / fetches the ticker's historical
     daily closing prices, parses them, and prints a clean ``date,close`` CSV
     to stdout between sentinel fences.

We poll the request until it completes, pull the CSV out of the sandbox's
stdout (``sandbox_results.results[].stdout``), save it, and render the line
chart locally with matplotlib.

Why this shape?
- The ``sandbox`` tool is rejected on the synchronous/streaming path
  ("streaming failed: ... unknown tool"); it must run with ``background: true``
  and be polled by id. This script always does that.
- ``sandbox_results`` carries only text (code/stdout/stderr) — there is no
  binary artifact channel — so the chart is rendered on this side, and you
  also keep a reusable ``.csv``.
- Top-level ``finance_search`` returns only the latest quote (no history) on
  the current deployment, so the price *series* is gathered from inside the
  sandbox. Because that relies on third-party web data, it is best-effort:
  the script retries the whole call a few times until it gets a usable CSV.

The Agent API is called over **raw HTTP** (no SDK) so the request body — and
the sandbox tool in it — is fully visible, and the endpoint is configurable
via ``--base-url`` / ``PERPLEXITY_BASE_URL``.

Docs:
- Agent API:      https://docs.perplexity.ai/docs/agent-api/quickstart
- sandbox:        https://docs.perplexity.ai/docs/agent-api/tools/sandbox
- finance_search: https://docs.perplexity.ai/docs/agent-api/tools/finance-search
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_BASE_URL = "https://api.perplexity.ai"
RESPONSES_PATH = "/v1/responses"

CSV_START = "===CSV_START==="
CSV_END = "===CSV_END==="


# Friendly --period values mapped to a natural-language phrase the model can
# act on. Anything not in this map is passed through verbatim.
PERIOD_PHRASES: Dict[str, str] = {
    "1mo": "the past 1 month",
    "3mo": "the past 3 months",
    "6mo": "the past 6 months",
    "1y": "the past 1 year",
    "2y": "the past 2 years",
    "5y": "the past 5 years",
}


SYSTEM_PROMPT = f"""You run inside a Python sandbox that includes the
`perplexity` SDK (web search and URL fetch) plus pandas and the standard
library. Your job is to produce a CSV of a stock's daily closing prices.

Approach:
- Use the perplexity SDK to obtain the daily closing prices: search the web
  and/or fetch a historical-price page that exposes a clean date/close table.
- If a source fails or is rate-limited, try a different one. Do not give up
  after a single failure.
- Print ONLY the final CSV to stdout, wrapped exactly between these fences:
      {CSV_START}
      <csv here>
      {CSV_END}
  Header must be `date,close`; one row per trading day; sorted ascending by
  date; dates as YYYY-MM-DD; close as a plain number. Put no logs, debug
  output, or commentary inside the fences.

Never fabricate or interpolate prices — use only values you actually
retrieved."""


PROMPT_TEMPLATE = (
    "Produce the daily closing-price CSV for {ticker} over {period_phrase}. "
    "Print it to stdout between the {start} / {end} fences."
)


# ---------------------------------------------------------------------------
# API key + HTTP helpers
# ---------------------------------------------------------------------------
def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Find the Perplexity API key.

    Order: explicit argument, ``PERPLEXITY_API_KEY`` / ``PPLX_API_KEY`` env
    vars, a ``.pplx_api_key`` file, or a ``KEY=value`` line in a local
    ``.env`` (``PERPLEXITY_API_KEY`` or ``PPLX_API_KEY``).
    """
    if api_key:
        return api_key
    for var in ("PERPLEXITY_API_KEY", "PPLX_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    for candidate in (".pplx_api_key", "pplx_api_key"):
        path = Path(candidate)
        if path.exists():
            return path.read_text().strip()
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in ("PERPLEXITY_API_KEY", "PPLX_API_KEY"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError(
        "API key not found. Set PERPLEXITY_API_KEY, pass --api-key, or add it "
        "to a .pplx_api_key / .env file."
    )


def _request(
    method: str, url: str, key: str, body: Optional[dict], timeout: int
) -> Tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "api-cookbook-finance-chart-sandbox/1.0",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read().decode())
        except Exception:  # noqa: BLE001
            return err.code, {"error": {"message": err.reason}}


def _poll(base_url: str, key: str, response_id: str, deadline: float) -> dict:
    """Poll a background response until terminal status (resilient to 5xx)."""
    url = f"{base_url}{RESPONSES_PATH}/{response_id}"
    body: dict = {"status": "in_progress"}
    while body.get("status") in ("queued", "in_progress"):
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for the sandbox response.")
        time.sleep(3)
        status, body = _request("GET", url, key, None, timeout=60)
        if status >= 500:
            # Transient server error mid-poll — keep waiting.
            body = {"status": "in_progress"}
    return body


def run_sandbox_request(
    base_url: str,
    key: str,
    ticker: str,
    period_phrase: str,
    model: str,
    max_steps: int,
    poll_timeout: int,
) -> dict:
    """Submit one background Agent API call and poll it to completion."""
    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": PROMPT_TEMPLATE.format(
            ticker=ticker.upper(),
            period_phrase=period_phrase,
            start=CSV_START,
            end=CSV_END,
        ),
        "tools": [{"type": "sandbox"}],
        "background": True,
        "max_output_tokens": 4096,
        "max_steps": max_steps,
    }
    status, body = _request(
        "POST", f"{base_url}{RESPONSES_PATH}", key, payload, timeout=120
    )
    if status >= 400:
        raise RuntimeError(f"Agent API error {status}: {body.get('error', body)}")
    if not body.get("id"):
        raise RuntimeError(f"Unexpected response (no id): {body}")
    return _poll(base_url, key, body["id"], deadline=time.time() + poll_timeout)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _sandbox_stdout(response: dict) -> str:
    """Concatenate stdout from every sandbox execution result."""
    chunks: List[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "sandbox_results":
            continue
        # Real shape nests executions under `results`; tolerate a flat shape.
        results = item.get("results")
        if results:
            for res in results:
                if res.get("stdout"):
                    chunks.append(res["stdout"])
        elif item.get("stdout"):
            chunks.append(item["stdout"])
    return "\n".join(chunks)


def _message_text(response: dict) -> str:
    """Concatenate assistant ``output_text`` blocks."""
    chunks: List[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if block.get("type") == "output_text" and block.get("text"):
                chunks.append(block["text"])
    return "\n".join(chunks)


def extract_csv(response: dict) -> Optional[str]:
    """Find the fenced CSV in the sandbox stdout, then the message text.

    Returns the CSV body (without fences), or None if nothing usable is found.
    """
    fence = re.compile(
        re.escape(CSV_START) + r"\s*(.*?)\s*" + re.escape(CSV_END), re.S
    )
    for haystack in (_sandbox_stdout(response), _message_text(response)):
        match = fence.search(haystack)
        if match and match.group(1).strip():
            return match.group(1).strip()
    # Fallback: a fenced ```csv block in the message.
    block = re.search(r"```csv\s*(.*?)```", _message_text(response), re.S)
    if block:
        lines = block.group(1).strip().splitlines()
        if lines and "date" in lines[0].lower():
            return block.group(1).strip()
    return None


def parse_csv(csv_text: str) -> Tuple[List[datetime], List[float]]:
    """Parse `date,close` CSV text into parallel lists, sorted by date."""
    reader = csv.DictReader(csv_text.splitlines())
    if not reader.fieldnames:
        raise RuntimeError("Empty CSV.")
    cols = {name.strip().lower(): name for name in reader.fieldnames}
    if "date" not in cols or "close" not in cols:
        raise RuntimeError(
            f"CSV missing date/close columns; got {reader.fieldnames}."
        )

    rows: List[Tuple[datetime, float]] = []
    for row in reader:
        raw_date = (row.get(cols["date"]) or "").strip()
        raw_close = (row.get(cols["close"]) or "").strip()
        if not raw_date or not raw_close:
            continue
        try:
            day = datetime.strptime(raw_date[:10], "%Y-%m-%d")
            close = float(raw_close.replace(",", "").replace("$", ""))
        except ValueError:
            continue
        rows.append((day, close))

    if len(rows) < 2:
        raise RuntimeError("Fewer than 2 valid date,close rows parsed.")
    rows.sort(key=lambda r: r[0])
    return [r[0] for r in rows], [r[1] for r in rows]


def render_chart(
    dates: List[datetime],
    closes: List[float],
    ticker: str,
    period_label: str,
    png_path: Path,
) -> None:
    """Render a closing-price line chart to ``png_path``."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, closes, color="#1f77b4", linewidth=1.6)
    ax.fill_between(dates, closes, min(closes), color="#1f77b4", alpha=0.08)
    ax.set_title(f"{ticker.upper()} closing price — {period_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close (USD)")
    ax.grid(True, linestyle="--", alpha=0.4)

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def sandbox_invocations(response: dict) -> int:
    details = (response.get("usage") or {}).get("tool_calls_details") or {}
    return (details.get("sandbox") or {}).get("invocation", 0) or 0


def total_cost(response: dict) -> Optional[Tuple[float, str]]:
    cost = (response.get("usage") or {}).get("cost")
    if not isinstance(cost, dict) or cost.get("total_cost") is None:
        return None
    return float(cost["total_cost"]), cost.get("currency", "USD")


def build_period_phrase(
    period: Optional[str], start: Optional[str], end: Optional[str]
) -> Tuple[str, str]:
    """Return (period_label, natural-language phrase) for prompts/filenames."""
    if start or end:
        label = f"{start}..{end}"
        if start and end:
            phrase = f"the period from {start} to {end}"
        elif start:
            phrase = f"the period since {start}"
        else:
            phrase = f"the period up to {end}"
        return label, phrase
    period = period or "6mo"
    return period, PERIOD_PHRASES.get(period, f"the past {period}")


def fetch_price_series(
    base_url: str,
    key: str,
    ticker: str,
    period_phrase: str,
    model: str,
    attempts: int,
    max_steps: int,
    poll_timeout: int,
    on_attempt=None,
) -> Tuple[List[datetime], List[float], str, dict]:
    """Run up to ``attempts`` background sandbox calls until a usable CSV parses.

    Returns ``(dates, closes, csv_text, response)``. Raises ``RuntimeError`` if
    no attempt yields a parseable ``date,close`` CSV. ``on_attempt(n, total,
    note)`` is an optional progress callback (``note`` is None at the start of
    an attempt, or a short failure reason).
    """
    response: dict = {}
    for attempt in range(1, attempts + 1):
        if on_attempt:
            on_attempt(attempt, attempts, None)
        try:
            response = run_sandbox_request(
                base_url, key, ticker, period_phrase, model, max_steps, poll_timeout
            )
        except (RuntimeError, TimeoutError) as err:
            if on_attempt:
                on_attempt(attempt, attempts, str(err))
            continue

        if response.get("status") == "failed":
            if on_attempt:
                on_attempt(attempt, attempts, f"request failed: {response.get('error')}")
            continue

        candidate = extract_csv(response)
        if not candidate:
            if on_attempt:
                on_attempt(attempt, attempts, "no fenced CSV in output")
            continue
        try:
            dates, closes = parse_csv(candidate)
        except RuntimeError as err:
            if on_attempt:
                on_attempt(attempt, attempts, f"unusable CSV: {err}")
            continue
        return dates, closes, candidate, response

    raise RuntimeError(
        f"Could not obtain a usable price CSV from the sandbox after "
        f"{attempts} attempt(s). Sandbox data fetching is best-effort "
        "(third-party sources rate-limit); try more attempts or rerun."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a stock's closing-price history using the Perplexity Agent "
            "API sandbox tool (background task), rendered locally."
        )
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL, MSFT, NVDA.")
    parser.add_argument(
        "--period",
        default="6mo",
        help="Lookback window: 1mo, 3mo, 6mo (default), 1y, 2y, 5y, ...",
    )
    parser.add_argument("--start", help="Start date YYYY-MM-DD (overrides --period).")
    parser.add_argument("--end", help="End date YYYY-MM-DD (use with --start).")
    parser.add_argument("--model", default="openai/gpt-5.5", help="Agent API model.")
    parser.add_argument("--out-dir", default=".", help="Directory for the CSV and PNG.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Max background calls to try until a usable CSV comes back "
        "(each is a separate sandbox session). Default 3.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=25, help="Agent max_steps per attempt."
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=300,
        help="Seconds to poll each background call before giving up.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PERPLEXITY_BASE_URL", DEFAULT_BASE_URL),
        help="Agent API base URL (or set PERPLEXITY_BASE_URL).",
    )
    parser.add_argument("--api-key", help="Perplexity API key.")
    parser.add_argument(
        "--keep-json", action="store_true", help="Write the raw response JSON."
    )
    args = parser.parse_args()

    try:
        key = resolve_api_key(args.api_key)
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    ticker = args.ticker.upper()
    period_label, period_phrase = build_period_phrase(
        args.period, args.start, args.end
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{ticker}_{period_label}".replace("..", "_to_")
    csv_path = out_dir / f"{slug}.csv"
    png_path = out_dir / f"{slug}.png"

    def _log(attempt: int, total: int, note: Optional[str]) -> None:
        if note is None:
            print(
                f"[attempt {attempt}/{total}] Asking the sandbox to fetch "
                f"{ticker} closing prices over {period_phrase}...",
                file=sys.stderr,
            )
        else:
            print(f"  {note}", file=sys.stderr)

    try:
        dates, closes, csv_text, response = fetch_price_series(
            args.base_url, key, ticker, period_phrase, args.model,
            args.attempts, args.max_steps, args.poll_timeout, on_attempt=_log,
        )
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 3

    if args.keep_json and response:
        (out_dir / f"{slug}.json").write_text(json.dumps(response, indent=2))

    csv_path.write_text(csv_text + "\n")
    render_chart(dates, closes, ticker, period_label, png_path)

    print(f"\nData points: {len(dates)} "
          f"({dates[0]:%Y-%m-%d} → {dates[-1]:%Y-%m-%d})")
    print(f"CSV:   {csv_path}")
    print(f"Chart: {png_path}")
    print(f"Sandbox invocations: {sandbox_invocations(response)}")
    cost = total_cost(response)
    if cost is not None:
        print(f"Cost: {cost[0]:.4f} {cost[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
