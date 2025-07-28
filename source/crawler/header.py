# crawler/headers.py
"""
User-agent, proxy, and screen-size utilities for request configuration
"""
import random
import json
import os

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
    "Mozilla/5.0 (X11; Linux x86_64)...",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)... Safari/605.1.15"
]

SCREEN_SIZES = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1280, "height": 800},
    {"width": 1440, "height": 900}
]

PROXIES = []
try:
    with open("proxy_pool.json", "r") as f:
        PROXIES = json.load(f)
except Exception:
    PROXIES = []

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_random_proxy():
    return random.choice(PROXIES) if PROXIES else None

def get_random_screen_size():
    return random.choice(SCREEN_SIZES)
