"""Exercise new-user creation and profile switching in the actual browser."""
import json
import os
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.getenv("APPBOOK_URL", "http://127.0.0.1:8031")
CHROME = "/Users/richmondalake/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
checks, errors = [], []


def check(name, condition):
    assert condition, name
    checks.append(name)
    print("PASS:", name, flush=True)


original = requests.get(BASE + "/api/state", timeout=180).json()
created = None
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        page = browser.new_page(viewport={"width": 1540, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(BASE, wait_until="networkidle")
        check("Traveler selector shows the current saved traveler", page.locator("#traveler-profile").input_value() == original["scope"]["user"])
        page.locator("#new-user").click()
        check("New user dialog describes a fresh profile", page.locator("#user-dialog").is_visible())
        page.locator("#new-user-label").fill("New traveler · browser validation")
        with page.expect_response(lambda response: response.url.endswith("/api/users") and response.request.method == "POST") as response:
            page.locator("#create-user").click()
        created = response.value.json()
        page.wait_for_function("(id) => document.querySelector('#traveler-profile').value === id", arg=created["scope"]["user"])
        check("Fresh UI has no previous conversation, name or trip", not page.locator("#profile-name").count() and not page.locator(".message").count() and created["session"] is None and created["preferences"] == {})
        page.screenshot(path=str(ROOT / "new_user.png"), full_page=True)
        page.locator("#new-conversation").click()
        page.locator("#keep-trip").uncheck()
        with page.expect_response(lambda response: response.url.endswith("/api/conversations")) as response:
            page.locator("#start-conversation").click()
        conversation = response.value.json()
        check("New conversation retains the selected user", conversation["scope"]["user"] == created["scope"]["user"] and conversation["scope"]["thread"] != created["scope"]["thread"])
        page.locator("#traveler-profile").select_option(original["scope"]["user"])
        page.wait_for_function("(id) => state.scope.user === id && !actionPending", arg=original["scope"]["user"])
        restored = requests.get(BASE + "/api/state", timeout=180).json()
        check("Switching back restores the original profile, trip and conversation", all(restored[key] == original[key] for key in ["scope", "session", "preferences", "conversation"]))
        check("Original entity name reappears in the active trip card", page.locator("#profile-name").inner_text().lower() == original["preferences"]["name"]["value"].lower())
        page.locator("#traveler-profile").select_option(created["scope"]["user"])
        page.wait_for_function("(id) => state.scope.user === id && !actionPending", arg=created["scope"]["user"])
        check("Returning to new traveler restores its last thread", requests.get(BASE + "/api/state", timeout=90).json()["scope"] == conversation["scope"])
        page.locator("#traveler-profile").select_option(original["scope"]["user"])
        page.wait_for_function("(id) => state.scope.user === id && !actionPending", arg=original["scope"]["user"])
        page.reload(wait_until="networkidle")
        check("Selected traveler survives a page reload", page.locator("#traveler-profile").input_value() == original["scope"]["user"])
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(BASE, wait_until="networkidle")
        check("User controls fit the mobile viewport", mobile.locator("#new-user").is_visible() and mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))
        check("User interactions have no browser runtime errors", not errors)
        browser.close()
finally:
    response = requests.post(BASE + "/api/users/select", json={"user_id": original["scope"]["user"]}, timeout=180)
    response.raise_for_status()

(ROOT / "users_browser_validation.json").write_text(json.dumps({"status": "passed", "checks": checks, "errors": errors, "created_user": created["scope"]["user"]}, indent=2))
