# freelance_search

`freelance_search` is a small Windows service that searches for freelance jobs
matching a personal AI/ML profile once a day and posts a ranked shortlist to
Discord.

The project is designed around local execution:

- SerpApi Google Jobs is the primary source, limited to 3–5 queries per run.
- RemoteOK, Jobicy, Arbeitnow, and We Work Remotely are no-key fallback sources.
- Ollama performs scoring locally; no cloud LLM is used.
- SQLite permanently records listings shown in a digest.
- The Discord bot is owner-only and does not apply for jobs or write bids.

## Setup

1. Install Python 3.11 or newer.
2. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev]"
   ```

3. Copy `.env.example` to `.env` and fill in the Discord and SerpApi values.
4. Confirm Ollama is running locally and that `OLLAMA_MODEL` is available.
5. Run the test and lint checks:

   ```powershell
   pytest
   ruff check .
   ```

The source adapters are implemented behind one normalized listing interface.
The scoring, scheduler, pipeline orchestration, and Discord commands are
delivered in later milestones.

## Digest behavior

The daily message contains at most five ranked listings and stays within
Discord's 2,000-character message limit. A listing is marked as shown only
after Discord accepts the digest, so a failed delivery does not consume jobs.

## Live tests

Tests that need network access or a local Ollama instance will use the
`live` pytest marker and are excluded from the default CI run:

```powershell
pytest -m live
```

## License

MIT. See [LICENSE](LICENSE).
