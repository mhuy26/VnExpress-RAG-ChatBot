# crawler/stealth.py
"""
Stealth utilities for reducing detection in Playwright
"""

def add_stealth_scripts(page):
    """Inject stealth JavaScript overrides into the page"""
    stealth_js = """
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
    });

    Object.defineProperty(navigator, 'plugins', {
        get: () => [1, 2, 3, 4, 5],
    });

    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
    });

    window.chrome = { runtime: {} };
    
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
    """
    page.add_init_script(stealth_js)
