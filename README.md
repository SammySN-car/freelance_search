# freelance_search

[![CI](https://github.com/SammySN-car/freelance_search/actions/workflows/ci.yml/badge.svg)](https://github.com/SammySN-car/freelance_search/actions/workflows/ci.yml)

`freelance_search` is a daily opportunity scout that anyone can point at
their own career stage: it searches SerpApi Google Jobs and free remote-job
feeds for freelance work, internships, and entry-level jobs, scores each new
listing with a local Ollama Gemma model against **your** profile, and posts a
ranked list of up to 25 links to Discord each morning. A job is persisted as
shown only after Discord confirms delivery, so the same job is never shown
twice.

Everything it optimizes for comes from your own configuration — there is no
hardcoded location or seniority bias in the code.

## Quickstart

Requirements: Windows, Python 3.11+, and [Ollama](https://ollama.com/) running
locally.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ollama pull gemma3:4b
Copy-Item .env.example .env
# Edit .env: Discord + SerpApi values, then your PROFILE,
# LOCATION, and EXPERIENCE_LEVEL (see Customization below).
python -m freelance_search
```

## Customization

The whole pipeline — search queries, ranking, and judging — runs from the
details each person puts in their own `.env`:

| Setting | What it does |
|---|---|
| `PROFILE` | Free text: your skills, what you want, and `SKIP:` lines for what you never want. Embedded for ranking and shown to the judge. |
| `LOCATION` | Your location. Added to every Google Jobs query (SerpApi's native `location` parameter), and listings that state a *different* location are hard-dropped from the digest (jobs with no stated location or marked remote always pass). Leave empty for worldwide/remote-only. While the active level is `beginner`/`intermediate`, listings with senior/staff/lead/principal/director/head in the title are also dropped. |
| `EXPERIENCE_LEVEL` | e.g. `beginner`, `student`, `senior`. Used only when `GITHUB_URL` is empty. The judge scores opportunities that fit this level higher and roles demanding senior expertise lower. |
| `GITHUB_URL` | Optional GitHub profile. The tool downloads your actual code and has the local model detect your level from it — no typed label. Re-checked every `GITHUB_REFRESH_DAYS` (default 2) and cached in the database; falls back to `EXPERIENCE_LEVEL` when empty or unreachable. |
| `SERPAPI_QUERIES` | 3–8 JSON-quoted search strings. Mix freelance, internship, and job queries for your field. |
| `DIGEST_SIZE` | How many links appear per digest (1–40, default 25). |
| `MAX_TO_SCORE` | How many highest-similarity listings reach the local judge per run (default 50). |
| `MIN_SCORE` | Judge scores below this are excluded from the digest (0–10, default 5.0). |

No keyword filtering is used anywhere: retrieval ranks everything by embedding
similarity, and the local model judges the finalists against your profile.

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

- **Fetch:** SerpApi Google Jobs is primary (with your `LOCATION` on every
  query); RemoteOK, Jobicy, Arbeitnow, and We Work Remotely RSS are no-key
  fallback adapters.
- **Dedup:** SQLite stores listings and permanently shown IDs/URLs.
- **Retrieve:** Ollama embeddings compare the profile with each listing's title
  and description; cosine similarity is persisted without storing vectors.
- **Rerank:** The highest-similarity candidates are judged by local Gemma
  against your composed profile (profile + location + experience level), which
  must return strict JSON with a numeric score from 0–10. No listing is dropped
  by keyword matching.
- **Digest:** Scores below `MIN_SCORE` are removed; results are sorted
  deterministically and capped at `DIGEST_SIZE` (default 25). The digest is a
  numbered list of masked links — titles are visible, long tracking URLs are
  not — and splits across embeds when one would overflow.
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

## GitHub level detection

Instead of typing an experience level, point `GITHUB_URL` at your profile. Every
`GITHUB_REFRESH_DAYS` (default: 2) the tool:

1. lists all your public repos (GitHub's free API, no token needed),
2. downloads each repo's source archive and extracts real code files
   (forks and empty repos skipped; ~400 lines per repo, whole batch capped to
   what the local model can hold, newest repos first),
3. asks the local model to read the code and name a level, citing specific
   repos/files as evidence.

The verdict (level + 2-line evidence summary) is cached in the database and
becomes part of the profile the job judge scores against. Between checks it is
read from cache. If GitHub is unreachable, the last cached verdict is used; if
there is none, `EXPERIENCE_LEVEL` applies. `/status` shows which source is
active and what the verdict was.

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

The SerpApi free tier provides 250 searches per month. The default
configuration uses five queries per daily run (configurable between three and
eight): 5 × 30 = 150 searches, keeping the service within the monthly
allowance. Failed searches are not counted by SerpApi.

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
highest-similarity listings reach the slower Gemma judge (default: `50`), and
`DIGEST_SIZE` controls how many judged listings appear as links in the digest
(default: `25`). Embedding and judge results are persisted, so a listing is
embedded and judged at most once per lifetime unless its database is removed.

## Scope

This project does not write cover letters or bids, auto-apply to jobs, scrape
Upwork/Fiverr/LinkedIn, use cloud LLMs, or provide a web UI.

## License

MIT. See [LICENSE](LICENSE).
