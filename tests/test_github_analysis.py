import io
import tarfile

import httpx
import pytest
import respx

from freelance_search.github_analysis import (
    API_ROOT,
    CODELOAD_ROOT,
    GithubAnalyzer,
    GithubError,
    build_code_digest,
    extract_repo_code,
    fetch_repos,
    github_username,
    parse_verdict,
    usable_repos,
)
from freelance_search.models import GithubVerdict


def make_repo(name, **overrides):
    repo = {
        "name": name,
        "fork": False,
        "size": 10,
        "language": "Python",
        "description": "demo",
        "stargazers_count": 0,
        "default_branch": "main",
    }
    repo.update(overrides)
    return repo


def make_tarball(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"repo-main/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_username_extracted_from_profile_url():
    assert github_username("https://github.com/octocat") == "octocat"
    assert github_username("https://github.com/octocat/") == "octocat"
    assert github_username("  https://github.com/octocat/repo  ") == "octocat"


def test_username_rejects_non_github_urls():
    with pytest.raises(GithubError):
        github_username("")
    with pytest.raises(GithubError):
        github_username("https://gitlab.com/octocat")


def test_usable_repos_drop_forks_and_empty():
    repos = [
        make_repo("keep"),
        make_repo("forked", fork=True),
        make_repo("empty", size=0),
        make_repo(""), 
    ]

    kept = usable_repos(repos)

    assert [repo["name"] for repo in kept] == ["keep"]


def test_extract_repo_code_reads_real_sources():
    archive = make_tarball(
        {
            "main.py": "def solve():\n    return 42\n",
            "node_modules/dep/index.js": "junk();",
            "app.min.js": "var a=1;",
            "README.md": "# not code",
        }
    )

    code = extract_repo_code(archive, "demo")

    assert "def solve():" in code
    assert "demo/" in code
    assert "junk();" not in code
    assert "var a=1;" not in code


def test_extract_repo_code_caps_repo_lines():
    big = make_tarball({"main.py": "x = 1\n" * 5000})

    code = extract_repo_code(big, "demo")

    assert code.count("x = 1") <= 400


def test_parse_verdict_valid_and_fenced():
    assert parse_verdict('{"level": "beginner", "summary": "cited repos"}') == {
        "level": "beginner",
        "summary": "cited repos",
    }
    fenced = '```json\n{"level": "Senior", "summary": "ok"}\n```'
    assert parse_verdict(fenced)["level"] == "senior"


def test_parse_verdict_rejects_bad_payloads():
    for raw in ("not json", '{"level": "wizard", "summary": "x"}', '{"level": "beginner"}'):
        with pytest.raises(GithubError):
            parse_verdict(raw)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_repos_paginates_and_filters():
    page1 = [make_repo(f"r{n}") for n in range(100)]
    page1[0] = make_repo("forked", fork=True)
    respx.get(f"{API_ROOT}/users/octocat/repos").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=[make_repo("last")]),
        ]
    )

    async with httpx.AsyncClient() as client:
        repos = await fetch_repos(client, "octocat")

    assert len(repos) == 100
    assert repos[-1]["name"] == "last"
    assert all(not repo["fork"] for repo in repos)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_repos_raises_on_http_error():
    respx.get(f"{API_ROOT}/users/octocat/repos").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        with pytest.raises(GithubError):
            await fetch_repos(client, "octocat")


@pytest.mark.asyncio
@respx.mock
async def test_build_code_digest_covers_every_repo_within_budget():
    repos = [make_repo(f"repo{n}") for n in range(5)]
    tarball = make_tarball({"main.py": "print('hi')\n" * 30})
    for repo in repos:
        respx.get(f"{CODELOAD_ROOT}/octocat/{repo['name']}/tar.gz/main").mock(
            return_value=httpx.Response(200, content=tarball)
        )

    async with httpx.AsyncClient() as client:
        digest, repos_read = await build_code_digest(client, "octocat", repos)

    for repo in repos:
        assert f"## Repo: {repo['name']}" in digest
    assert repos_read == 5
    assert "print('hi')" in digest


@pytest.mark.asyncio
@respx.mock
async def test_build_code_digest_still_lists_repos_it_cannot_read():
    respx.get(f"{CODELOAD_ROOT}/octocat/broken/tar.gz/main").mock(
        return_value=httpx.Response(404)
    )

    async with httpx.AsyncClient() as client:
        digest, repos_read = await build_code_digest(client, "octocat", [make_repo("broken")])

    assert "## Repo: broken" in digest
    assert "(no readable source)" in digest
    assert repos_read == 0


@pytest.mark.asyncio
@respx.mock
async def test_analyzer_returns_verdict_end_to_end():
    tarball = make_tarball({"main.py": "x = 1\n"})
    respx.get(f"{API_ROOT}/users/octocat/repos").mock(
        return_value=httpx.Response(200, json=[make_repo("demo")])
    )
    respx.get(f"{CODELOAD_ROOT}/octocat/demo/tar.gz/main").mock(
        return_value=httpx.Response(200, content=tarball)
    )
    respx.post("http://ollama.test/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": '{"level": "beginner", '
                '"summary": "simple scripts in demo/main.py; small scope overall"}'
            },
        )
    )

    async with httpx.AsyncClient() as client:
        verdict = await GithubAnalyzer(
            "https://github.com/octocat", "http://ollama.test", "gemma3:4b"
        ).analyze(client)

    assert isinstance(verdict, GithubVerdict)
    assert verdict.level == "beginner"
    assert verdict.repos_total == 1
    assert verdict.repos_read == 1
    assert "demo" in verdict.summary


def test_github_verdict_freshness_comparator(tmp_path):
    from freelance_search.discord_bot import BotService

    service = BotService.__new__(BotService)

    class FakeSettings:
        github_refresh_days = 2

    service.settings = FakeSettings()
    from datetime import datetime, timedelta

    fresh = GithubVerdict("beginner", "s", 1, 1, datetime.now().isoformat())
    stale = GithubVerdict(
        "beginner", "s", 1, 1, (datetime.now() - timedelta(days=3)).isoformat()
    )
    broken = GithubVerdict("beginner", "s", 1, 1, "not-a-date")

    assert service._github_is_fresh(fresh) is True
    assert service._github_is_fresh(stale) is False
    assert service._github_is_fresh(broken) is False