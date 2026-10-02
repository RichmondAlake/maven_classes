"""Start and cooperatively stop a real experiment using the browser controls."""
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
    page.goto("http://127.0.0.1:8031/#tokenomics", wait_until="networkidle")
    page.wait_for_function(
        'document.querySelector("#db-status").textContent.includes("connected")'
    )
    for checkbox in page.locator("[name=experiment-agent]").all():
        checkbox.set_checked(checkbox.get_attribute("value") == "custom")
    for checkbox in page.locator("[name=experiment-option]").all():
        checkbox.uncheck()
    page.locator("#experiment-mode").select_option("custom")
    page.locator("#experiment-turns").fill("3")
    page.locator("#experiment-turns").dispatch_event("change")
    page.locator("#experiment-prompts").fill(
        "Briefly explain how prompt caching helps agent memory.\nExplain workflow memory briefly.\nExplain semantic cache briefly."
    )
    with page.expect_response(
        lambda r: r.url.endswith("/api/tokenomics"), timeout=60000
    ) as response:
        page.locator("#run-experiment").click()
    job = response.value.json()
    check(
        "Browser submits the selected agents, options, turn count and context budget",
        job["agents"] == ["custom"]
        and job["turns"] == 3
        and job["context_limit"] == 24000
        and not any(job["options"].values()),
    )
    page.wait_for_function(
        'document.querySelector("#experiment-progress").textContent.includes("Custom memory agent · turn 1")',
        timeout=120000,
    )
    check(
        "Running experiment disables duplicate starts",
        page.locator("#run-experiment").is_disabled(),
    )
    with page.expect_response(
        lambda r: r.url.endswith("/cancel"), timeout=60000
    ) as response:
        page.locator("#stop-experiment").click()
    check(
        "Stop control requests cooperative cancellation",
        response.value.json()["cancel_requested"],
    )
    page.wait_for_function(
        'document.querySelector("#experiment-progress").textContent.includes("cancelled")',
        timeout=120000,
    )
    observed = page.request.get(
        "http://127.0.0.1:8031/api/tokenomics/" + job["id"]
    ).json()
    check(
        "Cancellation preserves the current paid turn and stops later turns",
        observed["status"] == "cancelled"
        and len(observed["rows"]) == 1
        and observed["rows"][0]["status"] == "success"
        and observed["rows"][0]["provider_calls"] > 0,
    )
    check(
        "Finished controls re-enable a new run and render observed charts",
        page.locator("#run-experiment").is_enabled()
        and page.locator(".experiment-chart").count() == 4,
    )
    check("Experiment controls produce no browser JavaScript errors", not errors)
    browser.close()

(ROOT / "tokenomics_browser_validation.json").write_text(
    json.dumps(
        {"status": "passed", "checks": checks, "errors": errors, "job_id": job["id"]},
        indent=2,
    )
)
