import json, os, secrets
from getpass import getpass
from pathlib import Path
from werkzeug.security import generate_password_hash

if __name__ == "__main__":
    os.umask(0o077)
    directory = Path(__file__).resolve().parent / "instance"
    directory.mkdir(mode=0o700, exist_ok=True)
    target = directory / "hosted.json"
    password = getpass("Choose the classroom demo password (at least 8 characters): ")
    if not 8 <= len(password) <= 256 or password != getpass("Confirm password: "):
        raise SystemExit("Passwords must match and have 8 to 256 characters. Run again.")
    data = dict(secret=secrets.token_hex(32),password_hash=generate_password_hash(password))
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf8")
    temporary.replace(target)
    print("Username: demo. Your classroom password is ready.")
