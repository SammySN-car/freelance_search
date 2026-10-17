import io
import logging
import tarfile
from datetime import datetime
from pathlib import PurePosixPath

import httpx

from .models import GithubVerdict

LOGGER = logging.getLogger(__name__)

API_ROOT = "https://api.github.com"
CODELOAD_ROOT = "https://codeload.github.com"
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".ipynb",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}
SKIP_PARTS = {
    ".git",
    ".next",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "vendor",
    "venv",
    ".venv",
}
SKIP_SUFFIXES = (".lock", ".map", ".min.css", ".min.js", "_pb2.py", ".pb.go")
MAX_FILE_LINES = 200
MAX_REPO_LINES = 400
MAX_REPO_CHARS = 8000
MAX_TOTAL_CHARS = 60000
MAX_FILE_BYTES = 512_000
MAX_TARBALL_BYTES = 20_000_000
VALID_LEVELS = {"beginner", "intermediate", "senior"}


class GithubError(Exception):
    """Raised when GitHub data or the local verdict cannot be produced."""


def github_username(github_url: str) -> str:
    cleaned = github_url.strip().rstrip("/")
    if not cleaned or "github.com" not in cleaned:
        raise GithubError("GITHUB_URL must point at a github.com profile")
    path = cleaned.split("github.com", 1)[1].lstrip("/")
    username = path.split("/")[0]
    if not username:
        raise GithubError("GITHUB_URL must include a profile name")
    return username


def usable_repos(repos: list[dict]) -> list[dict]:
    """Keep the owner's own, non-empty repos; drop forks and empty shells."""
    return [
        repo
        for repo in repos
        if not repo.get("fork") and (repo.get("size") or 0) > 0 and repo.get("name")
    ]


async def fetch_repos(client: httpx.AsyncClient, username: str) -> list[dict]:
    repos: list[dict] = []
    page = 1
    while page <= 10:
        response = await client.get(
            f"{API_ROOT}/users/{username}/repos",
            params={"per_page": 100, "page": page, "sort": "updated", "type": "owner"},
        )
        if response.status_code != 200:
            raise GithubError(f"GitHub API returned {response.status_code}")
        batch = response.json()
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return usable_repos(repos)


def _skip_member(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if any(part in SKIP_PARTS for part in parts):
        return True
    lowered = name.lower()
    return lowered.endswith(SKIP_SUFFIXES)


def extract_repo_code(archive_bytes: bytes, repo_name: str) -> str:
    """Pull real source files out of a repo tarball, capped per repo."""
    chunks: list[str] = []
    lines_used = 0
    chars_used = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar:
                if lines_used >= MAX_REPO_LINES or chars_used >= MAX_REPO_CHARS:
                    break
                if not member.isfile() or member.size > MAX_FILE_BYTES:
                    continue
                if _skip_member(member.name):
                    continue
                if "." not in member.name:
                    continue
                extension = "." + member.name.rsplit(".", 1)[-1].lower()
                if extension not in SOURCE_EXTENSIONS:
                    continue
                fileobj = tar.extractfile(member)
                if fileobj is None:
                    continue
                text = fileobj.read(MAX_FILE_BYTES).decode("utf-8", errors="ignore")
                file_lines = text.splitlines()[:MAX_FILE_LINES]
                remaining_lines = MAX_REPO_LINES - lines_used
                remaining_chars = MAX_REPO_CHARS - chars_used
                block = "\n".join(file_lines)[:remaining_chars]
                block = "\n".join(block.splitlines()[:remaining_lines])
                if not block.strip():
                    continue
                relative = str(PurePosixPath(member.name).relative_to(*member.name.split("/")[:1]))
                chunks.append(f"--- {repo_name}/{relative} ---\n{block}")
                lines_used += len(block.splitlines())
                chars_used += len(block)
    except tarfile.TarError as exc:
        raise GithubError(f"could not read archive for {repo_name}: {exc}") from exc
    return "\n\n".join(chunks)


async def fetch_archive(client: httpx.AsyncClient, username: str, repo: dict) -> bytes:
    branch = repo.get("default_branch") or "main"
    url = f"{CODELOAD_ROOT}/{username}/{repo['name']}/tar.gz/{branch}"
    response = await client.get(url)
    if response.status_code != 200:
        raise GithubError(f"archive for {repo['name']} returned {response.status_code}")
    if len(response.content) > MAX_TARBALL_BYTES:
        raise GithubError(f"archive for {repo['name']} is too large to read")
    return response.content


def _repo_header(repo: dict) -> str:
    return (
        f"## Repo: {repo['name']}\n"
        f"Description: {repo.get('description') or 'none'}\n"
        f"Language: {repo.get('language') or 'unknown'} · "
        f"stars: {repo.get('stargazers_count', 0)}"
    )


async def build_code_digest(
    client: httpx.AsyncClient, username: str, repos: list[dict]
) -> tuple[str, int]:
    """Assemble a code digest covering every repo; read code where it fits."""
    parts: list[str] = []
    repos_read = 0
    total_chars = 0
    for repo in repos:
        header = _repo_header(repo)
        code = ""
        try:
            archive = await fetch_archive(client, username, repo)
            code = extract_repo_code(archive, repo["name"])
        except GithubError as exc:
            LOGGER.warning("Skipping repo %s: %s", repo["name"], exc)
        if code and total_chars + len(header) + len(code) <= MAX_TOTAL_CHARS:
            parts.append(f"{header}\n{code}")
            total_chars += len(header) + len(code)
            repos_read += 1
        else:
            note = "\n(code omitted: size budget reached)" if code else "\n(no readable source)"
            parts.append(header + note)
    return "\n\n".join(parts), repos_read


# Instruction goes AFTER the code: with a large code context, gemma3:4b follows
# recency and ignores format rules placed before it (verified locally).
GITHUB_PROMPT = (
    "CODE:\n{code}\n\n"
    "Now answer based ONLY on the code above, output exactly two lines:\n"
    "Level: beginner|intermediate|senior\n"
    "Evidence: <one line citing at least two specific repo or file names as proof>\n"
    "Judge what the code demonstrates: completeness, structure, testing, and complexity. "
    "Ignore stars and popularity. Output only those two lines, nothing else."
)


def parse_verdict(raw: str) -> dict[str, str]:
    """Read the two-line Level:/Evidence: verdict; ignore echoed prompt fragments."""
    level = ""
    summary = ""
    for line in raw.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("level:") and not level:
            level = stripped.split(":", 1)[1].strip().rstrip(".").lower()
        elif lowered.startswith("evidence:") and not summary:
            summary = stripped.split(":", 1)[1].strip()
    if level not in VALID_LEVELS:
        raise GithubError(f"verdict level invalid: {level!r}")
    if not summary:
        raise GithubError("verdict summary missing")
    return {"level": level, "summary": summary[:240]}


async def judge_code(
    client: httpx.AsyncClient,
    ollama_base_url: str,
    model: str,
    timeout_seconds: float,
    code_digest: str,
) -> dict[str, str]:
    try:
        response = await client.post(
            f"{ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": GITHUB_PROMPT.replace("{code}", code_digest),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise GithubError(f"local judge unreachable: {exc}") from exc
    payload = response.json()
    return parse_verdict(str(payload.get("response", "")))


class GithubAnalyzer:
    def __init__(
        self,
        github_url: str,
        ollama_base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.github_url = github_url
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def analyze(self, client: httpx.AsyncClient) -> GithubVerdict:
        username = github_username(self.github_url)
        repos = await fetch_repos(client, username)
        if not repos:
            raise GithubError(f"no usable public repos for {username}")
        digest, repos_read = await build_code_digest(client, username, repos)
        try:
            verdict = await judge_code(
                client, self.ollama_base_url, self.model, self.timeout_seconds, digest
            )
        except GithubError as exc:
            LOGGER.warning("verdict unparseable (%s); retrying once", exc)
            verdict = await judge_code(
                client, self.ollama_base_url, self.model, self.timeout_seconds, digest
            )
        return GithubVerdict(
            level=verdict["level"],
            summary=verdict["summary"],
            repos_total=len(repos),
            repos_read=repos_read,
            analyzed_at=datetime.now().isoformat(),
        )