"""
One-time session setup for tweety-ns.
======================================

TWO OPTIONS:

Option A — Cookie login (recommended, most reliable):
  1. Log into x.com in your browser
  2. Open DevTools (Cmd+Option+I on Mac, F12 on Windows)
  3. Go to Application > Cookies > https://x.com
  4. Find "auth_token" and copy its value
  5. Run:  python setup_session.py --cookie YOUR_AUTH_TOKEN_HERE

Option B — Username/password login:
  Run:  python setup_session.py
  (will prompt for username and password)

After either option, a session file (btc_session.tw_session) is
saved in the current directory. The pipeline loads it automatically.
"""

import sys
import asyncio
from tweety import TwitterAsync

SESSION_NAME = "btc_session"


async def cookie_login(auth_token: str):
    app = TwitterAsync(SESSION_NAME)
    await app.load_auth_token(auth_token)
    print(f"\n✓ Logged in as: {app.me}")
    print(f"✓ Session saved to: {SESSION_NAME}.tw_session")
    print("\nYou can now run: python diagnose.py")


async def password_login():
    username = input("Twitter/X username or email: ").strip()
    password = input("Password: ").strip()

    print("\nLogging in...")
    app = TwitterAsync(SESSION_NAME)

    try:
        await app.sign_in(username, password)
        print(f"\n✓ Logged in as: {app.me}")
        print(f"✓ Session saved to: {SESSION_NAME}.tw_session")
    except Exception as e:
        error_msg = str(e)
        if "action" in error_msg.lower():
            print(f"\nTwitter requires verification: {e}")
            extra = input("Enter the code: ").strip()
            try:
                await app.sign_in(username, password, extra=extra)
                print(f"\n✓ Logged in as: {app.me}")
                print(f"✓ Session saved to: {SESSION_NAME}.tw_session")
            except Exception as e2:
                print(f"\n✗ Login failed: {e2}")
        else:
            print(f"\n✗ Login failed: {e}")
            print("  Try the cookie method instead:")
            print("  python setup_session.py --cookie YOUR_AUTH_TOKEN")


async def main():
    print("=" * 50)
    print("  Twitter/X Session Setup")
    print("=" * 50)

    if len(sys.argv) >= 3 and sys.argv[1] == "--cookie":
        await cookie_login(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--cookie":
        token = input("Paste your auth_token cookie value: ").strip()
        await cookie_login(token)
    else:
        print("\nTip: cookie login is more reliable than password login.")
        print("Use: python setup_session.py --cookie YOUR_AUTH_TOKEN\n")
        await password_login()


if __name__ == "__main__":
    asyncio.run(main())