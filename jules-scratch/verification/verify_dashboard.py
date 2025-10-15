from playwright.sync_api import sync_playwright, expect


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        # Wait for the title to be visible to ensure the page has loaded
        expect(page.get_by_text("iEEG Data Analysis Dashboard")).to_be_visible(
            timeout=10000
        )

        page.screenshot(path="jules-scratch/verification/verification.png")
        browser.close()


if __name__ == "__main__":
    run()
