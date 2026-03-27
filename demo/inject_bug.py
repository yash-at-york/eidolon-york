"""
Eidolon - Bug Injector
Injects a specific bug into demo/app_test.py for the live conference demo.
Saves original content to demo/app_test.py.bak for restoration.

Usage:
    python demo/inject_bug.py          # inject the missing auth check bug
    python demo/inject_bug.py --restore  # restore original
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TARGET = Path("demo/app_test.py")
BACKUP = Path("demo/app_test.py.bak")

# The bug: remove the auth validation guard so verify_user_token is fully accessible
# without a valid JWT - a classic broken authentication pattern
INJECTED_BUG = '''
@app.post("/verify-token")
def verify_user_token(jwt_token: str, db_session: Session, user_id: int):
    # [Eidolon DEMO BUG INJECTED]: Auth check bypassed - any token accepted
    decoded_payload = decode_jwt(jwt_token)
    print(user_id)
    # Missing: if not decoded_payload: raise HTTPException(...)
    user_record = db_session.query(User).filter(User.id == user_id).first()
    return user_record
'''

ORIGINAL_FUNCTION = '''
@app.post("/verify-token")
def verify_user_token(jwt_token: str, db_session: Session, user_id: int):
    # BUG 1 (latent): Uses .last() instead of .first() - SQLAlchemy has no .last()
    # BUG 2 (injected by demo): Missing auth check before db query
    decoded_payload = decode_jwt(jwt_token)
    print(user_id)
    if not decoded_payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_record = db_session.query(User).filter(User.id == decoded_payload.id).last()
    return user_record
'''


def inject() -> None:
    content = TARGET.read_text()
    # Backup
    shutil.copy(TARGET, BACKUP)
    # Replace function
    new_content = content.replace(ORIGINAL_FUNCTION.strip(), INJECTED_BUG.strip())
    TARGET.write_text(new_content)
    print(f"Bug injected into {TARGET}")
    print(f"  Backup saved to {BACKUP}")
    print(f"  Watch the Eidolon daemon detect the structural change...")


def restore() -> None:
    if BACKUP.exists():
        shutil.copy(BACKUP, TARGET)
        BACKUP.unlink()
        print(f"Original restored from backup")
    else:
        print("No backup found - nothing to restore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        inject()
