# freelance_search

[![CI](https://github.com/SammySN-car/freelance_search/actions/workflows/ci.yml/badge.svg)](https://github.com/SammySN-car/freelance_search/actions/workflows/ci.yml)

`freelance_search` is a daily freelance scout for an AI/ML engineer: it
searches SerpApi Google Jobs and free remote-job feeds, scores each new listing
with a local Ollama Gemma model, and posts a ranked top-five shortlist to
Discord each morning. A job is persisted as shown only after Discord confirms
delivery, so the same job is never shown twice.

## Quickstart

Requirements: Windows, Python 3.11+, and [Ollama](https://ollama.com/) running
locally.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ollama pull gemma3:4b
Copy-Item .env.example .env
# Edit .env and fill in the Discord and SerpApi values.
python -m freelance_search
```

The default model is `gemma3:4b`. Set `OLLAMA_MODEL` in `.env` to use another
locally installed model. The default timezone is `Asia/Kolkata` and the
default digest time is `08:00`.

## Discord setup

1. Open the [Discord Developer Portal](https://discord.com/developers/applications)
   and create an application.
2. Open **Bot**, create the bot, and copy its token into `DISCORD_TOKEN`.
3. Enable the bot's **Message Content Intent** only if your Discord server
   requires it; this project uses slash commands and does not read message
   content.
4. Invite the bot to your server with the `bot` and `applications.commands`
   scopes and permission to view and send messages in the target channel.
5. In Discord, enable **Developer Mode** under **User Settings → Advanced**.
6. Right-click your own user and choose **Copy ID** for `DISCORD_OWNER_ID`.
7. Right-click the destination channel and choose **Copy ID** for
   `DISCORD_CHANNEL_ID`.

Keep `.env` private. It is ignored by Git and must never be committed.

## Commands and delivery

| Command / event | Who | Behavior |
|---|---|---|
| `/refresh` | Owner only | Runs the pipeline immediately, posts the digest to the configured channel, and returns a short confirmation; empty results are still acknowledged. |
| `/status` | Owner only | Shows the latest run, counts, source failures, SerpApi query usage, Ollama health, and scheduler state. |
| Scheduled daily digest | Automatic | Runs at `DIGEST_TIME` in `DIGEST_TIMEZONE`; posts nothing when there are no new matching jobs. |

Only one manual or scheduled pipeline can run at a time. A second invocation
receives `Run already in progress.`.

## Architecture

```text
fetch -> dedup -> retrieve (embeddings) -> rerank (local Gemma) -> digest -> deliver
                                                                            |
                                                                            +-> mark shown only after Discord confirms send
```

- **Fetch:** SerpApi Google Jobs is primary; RemoteOK, Jobicy, Arbeitnow, and
  We Work Remotely RSS are no-key fallback adapters.
- **Dedup:** SQLite stores listings and permanently shown IDs/URLs.
- **Retrieve:** Ollama embeddings compare the profile with each listing's title
  and description; cosine similarity is persisted without storing vectors.
- **Rerank:** The highest-similarity candidates are judged by local Gemma,
  which must return strict JSON with a numeric score from 0–10 and a one-line
  reason. No listing is dropped by keyword matching.
- **Digest:** Scores below `MIN_SCORE` are removed; results are sorted
  deterministically and capped at five in a Discord embed with masked links,
  keeping long tracking URLs out of the visible message.
- **Deliver:** the prepared message is sent first; only a confirmed send marks
  its listing IDs as shown.

Project layout:

```text
src/freelance_search/
├── config.py          # typed .env settings
├── db.py              # SQLite schema and persistence
├── discord_bot.py     # owner-only commands and daily scheduler
├── models.py          # normalized listings and result models
├── pipeline.py        # fetch -> dedup -> score -> digest orchestration
├── retriever.py       # batched local embeddings and cosine similarity
├── scoring/ollama.py  # local Ollama client and health check
└── sources/           # SerpApi, JSON API, and RSS adapters
tests/                 # hermetic tests plus one opt-in live Ollama test
```

## Testing and CI

The default test suite is hermetic: it uses mocked HTTP/source/scoring/Discord
boundaries and performs zero network requests.

```powershell
pytest
ruff check .
```

The live test requires a running local Ollama instance and is excluded from the
default suite:

```powershell
pytest -m live
```

GitHub Actions runs Ruff and the hermetic pytest suite on every push. No cloud
LLM calls are used by the application or CI.

## SerpApi quota

The SerpApi free tier provides 250 searches per month. The default configuration
uses approximately three queries per daily run (configurable between three and
five), keeping the service within the monthly allowance.

## Windows Task Scheduler

The Discord bot contains its own daily loop and is the recommended runtime.
For start-on-boot and restart-on-crash behavior, create a Windows Task
Scheduler task:

1. Set **Trigger** to **At startup**.
2. Set **Action** to start `python.exe` with argument
   `-m freelance_search`.
3. Set **Start in** to the repository directory.
4. Enable **Run whether user is logged on or not** if appropriate.
5. In **Settings**, enable **Restart the task if it fails** and choose a short
   restart interval.
6. Store the real `.env` in the repository directory with user-only access.

The service's daily loop is exact for fixed-offset timezones such as IST.
Timezones with daylight-saving transitions can drift after a DST transition;
restart the process or adjust the configured time if that matters for your
location.

## Retrieval tuning

Set `EMBEDDING_MODEL` to the Ollama embedding model you want to use
(default: `mxbai-embed-large`). `MAX_TO_SCORE` controls how many
highest-similarity listings reach the slower Gemma judge (default: `30`).
Embedding and judge results are persisted, so a listing is embedded and judged
at most once per lifetime unless its database is removed.

## Scope

This project does not write cover letters or bids, auto-apply to jobs, scrape
Upwork/Fiverr/LinkedIn, use cloud LLMs, or provide a web UI.

## License

MIT. See [LICENSE](LICENSE).
