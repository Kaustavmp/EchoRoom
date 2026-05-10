"""
EchoRoom – Meeting Bot Core (Joining Layer)
Supports: Google Meet, Zoom (web client), Microsoft Teams
Uses Playwright with stealth techniques to appear as a legitimate participant.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger("echoroom.bot")


class MeetingPlatform(str, Enum):
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    UNKNOWN = "unknown"


@dataclass
class BotConfig:
    bot_name: str = "EchoRoom Bot"
    headless: bool = True
    timeout_ms: int = 30_000          # 30 s for element waits
    join_wait_s: int = 5              # pause after clicking join
    stealth: bool = True
    google_account_email: Optional[str] = None
    google_account_password: Optional[str] = None
    user_data_dir: Optional[str] = None   # persist session / cookies
    extra_chromium_args: list = field(default_factory=list)


def detect_platform(url: str) -> MeetingPlatform:
    url = url.lower()
    if "meet.google.com" in url:
        return MeetingPlatform.GOOGLE_MEET
    if "zoom.us" in url or "zoom.com" in url:
        return MeetingPlatform.ZOOM
    if "teams.microsoft.com" in url or "teams.live.com" in url:
        return MeetingPlatform.TEAMS
    return MeetingPlatform.UNKNOWN


def _normalise_zoom_url(url: str) -> str:
    """
    Force Zoom into the web-client view and strip the desktop-app redirect.
    Appends ?pwd= if missing so the web client doesn't block joining.
    """
    # Convert  zoom.us/j/<id>  →  zoom.us/wc/<id>/join
    url = re.sub(r"zoom\.us/j/", "zoom.us/wc/", url)
    if "/join" not in url:
        url = url.rstrip("/") + "/join"
    if "pwd=" not in url:
        url += ("&" if "?" in url else "?") + "pwd="
    return url


class MeetingBot:
    """
    Browser-automation bot that joins a video meeting, captures live captions
    (DOM scraping), and yields speaker-tagged caption lines via an async queue.
    """

    def __init__(self, config: BotConfig | None = None):
        self.config = config or BotConfig()
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._platform: Optional[MeetingPlatform] = None
        self.caption_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    async def join(self, meeting_url: str) -> None:
        self._platform = detect_platform(meeting_url)
        logger.info("Detected platform: %s", self._platform)

        if self._platform == MeetingPlatform.ZOOM:
            meeting_url = _normalise_zoom_url(meeting_url)
            logger.info("Normalised Zoom URL: %s", meeting_url)

        await self._launch_browser()
        await self._navigate(meeting_url)
        await self._platform_join()
        self._running = True
        asyncio.create_task(self._scrape_captions_loop())
        logger.info("Bot is now in the meeting and scraping captions.")

    async def leave(self) -> None:
        self._running = False
        if self._page and not self._page.is_closed():
            await self._click_leave_button()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        logger.info("Bot has left the meeting.")

    # ------------------------------------------------------------------ #
    #  Browser launch                                                       #
    # ------------------------------------------------------------------ #

    async def _launch_browser(self) -> None:
        pw = await async_playwright().start()

        args = [
            "--use-fake-ui-for-media-stream",   # auto-grant mic/cam
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--window-size=1280,800",
        ] + self.config.extra_chromium_args

        launch_opts = dict(
            headless=self.config.headless,
            args=args,
        )

        if self.config.user_data_dir:
            # Persistent context reuses cookies / Google sign-in
            self._context = await pw.chromium.launch_persistent_context(
                self.config.user_data_dir,
                **launch_opts,
                permissions=["microphone", "camera"],
            )
            self._browser = None
        else:
            self._browser = await pw.chromium.launch(**launch_opts)
            self._context = await self._browser.new_context(
                permissions=["microphone", "camera"],
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
            )

        if self.config.stealth:
            await self._inject_stealth_scripts()

        self._page = await self._context.new_page()
        logger.debug("Browser launched (headless=%s)", self.config.headless)

    async def _inject_stealth_scripts(self) -> None:
        """Patch navigator properties so automation flags are hidden."""
        stealth_js = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
            window.chrome = { runtime: {} };
        """
        await self._context.add_init_script(stealth_js)

    # ------------------------------------------------------------------ #
    #  Navigation & platform-specific join flows                            #
    # ------------------------------------------------------------------ #

    async def _navigate(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded",
                              timeout=self.config.timeout_ms)

    async def _platform_join(self) -> None:
        dispatch = {
            MeetingPlatform.GOOGLE_MEET: self._join_google_meet,
            MeetingPlatform.ZOOM: self._join_zoom_web,
            MeetingPlatform.TEAMS: self._join_teams,
            MeetingPlatform.UNKNOWN: self._join_generic,
        }
        handler = dispatch.get(self._platform, self._join_generic)
        await handler()

    # --- Google Meet --------------------------------------------------- #

    async def _join_google_meet(self) -> None:
        page = self._page
        cfg = self.config

        # If a Google account is configured, sign in first
        if cfg.google_account_email and cfg.google_account_password:
            await self._google_sign_in()

        # Dismiss "Continue without signing in" if present
        try:
            btn = page.locator("text=Continue without signing in")
            await btn.click(timeout=5_000)
        except Exception:
            pass

        # Type bot name into the "What's your name?" field
        try:
            name_field = page.locator('input[placeholder="Your name"]')
            await name_field.fill(cfg.bot_name, timeout=8_000)
        except Exception:
            logger.debug("Name field not found (may already be signed in).")

        # Mute mic and cam before joining (good bot etiquette)
        for aria_label in ("Turn off microphone", "Turn off camera"):
            try:
                btn = page.locator(f'[aria-label="{aria_label}"]')
                if await btn.count() > 0:
                    await btn.click()
            except Exception:
                pass

        # Click "Ask to join" or "Join now"
        for label in ("Ask to join", "Join now", "Join"):
            try:
                btn = page.get_by_role("button", name=label)
                await btn.click(timeout=8_000)
                break
            except Exception:
                continue

        await asyncio.sleep(self.config.join_wait_s)

        # Turn on captions
        await self._enable_google_meet_captions()

    async def _google_sign_in(self) -> None:
        """Sign into Google before joining a Meet link."""
        page = self._page
        await page.goto("https://accounts.google.com/signin",
                        wait_until="domcontentloaded")
        await page.fill('input[type="email"]', self.config.google_account_email)
        await page.click("#identifierNext")
        await page.fill('input[type="password"]', self.config.google_account_password)
        await page.click("#passwordNext")
        await page.wait_for_load_state("networkidle", timeout=15_000)
        logger.info("Signed into Google account.")

    async def _enable_google_meet_captions(self) -> None:
        """Click the 'Turn on captions' button inside Meet."""
        selectors = [
            '[aria-label="Turn on captions"]',
            '[aria-label="Captions"]',
            'button[jsname="r8qRAd"]',
        ]
        for sel in selectors:
            try:
                await self._page.click(sel, timeout=5_000)
                logger.info("Captions enabled.")
                return
            except Exception:
                continue
        logger.warning("Could not enable captions automatically.")

    # --- Zoom Web Client ----------------------------------------------- #

    async def _join_zoom_web(self) -> None:
        page = self._page

        # Dismiss the "Open Zoom" dialog by staying in web client
        try:
            await page.click("text=join from your browser", timeout=8_000)
        except Exception:
            pass

        # Fill name
        try:
            await page.fill('input[placeholder="Your Name"]',
                            self.config.bot_name, timeout=8_000)
        except Exception:
            pass

        # Join button
        try:
            await page.click("text=Join", timeout=8_000)
        except Exception:
            pass

        await asyncio.sleep(self.config.join_wait_s)

    # --- Microsoft Teams ----------------------------------------------- #

    async def _join_teams(self) -> None:
        page = self._page

        # "Continue on this browser"
        try:
            await page.click("text=Continue on this browser", timeout=8_000)
        except Exception:
            pass

        # Name field
        try:
            await page.fill('[placeholder="Type your name"]',
                            self.config.bot_name, timeout=8_000)
        except Exception:
            pass

        # Join button
        try:
            await page.click("text=Join now", timeout=8_000)
        except Exception:
            pass

        await asyncio.sleep(self.config.join_wait_s)

    # --- Generic fallback --------------------------------------------- #

    async def _join_generic(self) -> None:
        logger.warning("Unknown platform – attempting generic join flow.")
        try:
            await self._page.fill('input[placeholder*="name" i]',
                                  self.config.bot_name, timeout=5_000)
        except Exception:
            pass
        for label in ("Join", "Ask to join", "Enter", "Start"):
            try:
                await self._page.get_by_role("button", name=label).click(
                    timeout=4_000)
                break
            except Exception:
                continue
        await asyncio.sleep(self.config.join_wait_s)

    # ------------------------------------------------------------------ #
    #  Caption scraping loop                                                #
    # ------------------------------------------------------------------ #

    _CAPTION_SELECTORS = {
        MeetingPlatform.GOOGLE_MEET: {
            "speaker": '[class*="zs7s8d"]',       # active speaker label
            "text": '[class*="CNusmb"]',           # caption text
        },
        MeetingPlatform.ZOOM: {
            "speaker": ".speaker-name",
            "text": ".caption-line",
        },
        MeetingPlatform.TEAMS: {
            "speaker": '[data-tid="captions-speaker-name"]',
            "text": '[data-tid="captions-text"]',
        },
    }

    async def _scrape_captions_loop(self) -> None:
        sel = self._CAPTION_SELECTORS.get(self._platform, {})
        speaker_sel = sel.get("speaker", "")
        text_sel = sel.get("text", "")
        last_text = ""

        while self._running:
            try:
                speaker = ""
                caption = ""

                if speaker_sel:
                    els = await self._page.query_selector_all(speaker_sel)
                    if els:
                        speaker = (await els[-1].inner_text()).strip()

                if text_sel:
                    els = await self._page.query_selector_all(text_sel)
                    if els:
                        caption = " ".join(
                            [(await e.inner_text()).strip() for e in els]
                        ).strip()

                if caption and caption != last_text:
                    last_text = caption
                    entry = {
                        "speaker": speaker or "Unknown",
                        "text": caption,
                        "timestamp": time.time(),
                    }
                    await self.caption_queue.put(entry)
                    logger.debug("Caption: %s", entry)

            except Exception as exc:
                logger.debug("Caption scrape error (non-fatal): %s", exc)

            await asyncio.sleep(0.5)   # poll every 500 ms

    # ------------------------------------------------------------------ #
    #  Leave                                                                #
    # ------------------------------------------------------------------ #

    async def _click_leave_button(self) -> None:
        for label in ("Leave call", "Leave", "End call", "Hang up"):
            try:
                await self._page.get_by_role("button", name=label).click(
                    timeout=4_000)
                return
            except Exception:
                continue
