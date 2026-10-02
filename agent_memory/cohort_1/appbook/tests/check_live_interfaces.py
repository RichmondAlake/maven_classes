"""Exercise paired interfaces and new conversations with real providers in Chromium."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
checks, errors = [], []


def check(name, condition):
    assert condition, name
    checks.append(name)
    print("PASS:", name, flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME)
    page = browser.new_page(viewport={"width": 1540, "height": 1100})
    page.on("pageerror", lambda error: errors.append(str(error)))

    def compare(section, query):
        page.goto("http://127.0.0.1:8031/#lab/" + section, wait_until="networkidle")
        page.locator("#shared-query").fill(query)
        with page.expect_response(
            lambda r: "/api/compare/" in r.url, timeout=240000
        ) as response:
            page.locator("#compare").click()
        value = response.value
        assert value.status == 200, value.text()[:250]
        page.wait_for_function(
            'document.querySelectorAll(".comparison-chat .message").length === 4'
        )
        return value.json()

    result = compare("3", "What are the TAP Portugal cabin baggage rules?")
    check(
        "One browser query renders two real reranking answers and rankings",
        bool(result["without"]["answer"])
        and bool(result["with"]["answer"])
        and page.locator("#context-without .source-result").count() > 0
        and page.locator("#context-with .source-result").count() > 0,
    )
    page.screenshot(path=str(ROOT / "reranking_comparison.png"), full_page=True)

    page.goto("http://127.0.0.1:8031/#lab/7", wait_until="networkidle")
    page.locator("#reset-semantic").click()
    cold = compare("7", "Explain how semantic cache helps agent memory.")
    warm = compare("7", "Explain how semantic cache helps agent memory.")
    check(
        "Semantic-cache browser control produces a cold miss and a warm hit",
        not cold["with"]["cache_hit"]
        and warm["with"]["cache_hit"]
        and warm["with"]["metrics"]["provider_calls"] == 0
        and warm["without"]["metrics"]["provider_calls"] == 1,
    )
    page.screenshot(path=str(ROOT / "semantic_cache_comparison.png"), full_page=True)

    compact = compare(
        "10", "What trip constraints and preferences did I ask you to remember?"
    )
    check(
        "Compaction browser displays actual answers, summary and verified archive savings",
        compact["archive"]["lossless_verified"]
        and not compact["active_history_modified"]
        and bool(page.locator("#context-with .memory-preview").inner_text())
        and "Exact reconstruction verified"
        in page.locator("#comparison-extra").inner_text(),
    )
    page.screenshot(path=str(ROOT / "compaction_comparison.png"), full_page=True)

    page.goto("http://127.0.0.1:8031/#assistant", wait_until="networkidle")
    page.locator("#new-conversation").click()
    with page.expect_response(
        lambda r: r.url.endswith("/api/conversations"), timeout=60000
    ) as response:
        page.locator("#start-conversation").click()
    fresh = response.value.json()
    page.wait_for_selector(".empty-chat")
    check(
        "Header button starts a real empty conversation while keeping the active trip",
        not fresh["conversation"]
        and bool(fresh["session"]["state"]["trip"])
        and page.locator(".empty-chat").is_visible(),
    )
    check("Live comparison interactions have no browser JavaScript errors", not errors)
    browser.close()

(ROOT / "live_interface_browser_validation.json").write_text(
    json.dumps({"status": "passed", "checks": checks, "errors": errors}, indent=2)
)
