from playwright.sync_api import sync_playwright
import logging

logger = logging.getLogger("web_agent")


class BrowserController:
    def __init__(self, headless=False):
        logger.debug("Starting Playwright (headless=%s)", headless)
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        logger.info("Browser launched")

    def navigate(self, url):
        logger.debug("navigate: %s", url)
        self.page.goto(url)
        logger.debug("navigation complete: %s", url)
        return f"Navigated to {url}"

    def fill_field(self, selector, value):
        logger.debug("fill_field: %s = %s", selector, value)
        self.page.fill(selector, value)
        return f"Filled {selector} with '{value}'"

    def click(self, selector):
        logger.debug("click: %s", selector)
        self.page.click(selector)
        return f"Clicked {selector}"

    def get_page_text(self):
        logger.debug("get_page_text called")
        text = self.page.inner_text("body")
        logger.debug("page text length: %d", len(text))
        return text[:3000]

    def upload_file(self, selector, filepath):
        logger.debug("upload_file: %s <= %s", selector, filepath)
        self.page.set_input_files(selector, filepath)
        return f"Uploaded file to {selector}"

    def screenshot(self, path="screenshot.png"):
        logger.debug("screenshot: %s", path)
        self.page.screenshot(path=path)
        return f"Screenshot saved to {path}"

    def close(self):
        logger.info("Closing browser")
        try:
            self.browser.close()
        finally:
            self.playwright.stop()