"""
Eidolon — Comprehensive Demo Application
A realistic microservice containing intentional bugs covering every error
category that the Eidolon diagnostic system is built to handle.

Error Type Coverage:
  BUG-01  NameError:         undefined reference (missing import)
  BUG-02  AttributeError:    wrong method on ORM object (.last() doesn't exist)
  BUG-03  TypeError:         wrong argument type passed to function
  BUG-04  KeyError:          dict access without .get() guard
  BUG-05  IndexError:        list access out of bounds
  BUG-06  ValueError:        int conversion of unparseable string
  BUG-07  ImportError:       module referenced but not importable
  BUG-08  AssertionError:    precondition assertion fails
  BUG-09  PermissionError:   missing auth guard before privileged operation
  BUG-10  TimeoutError:      no timeout set on external network call
  BUG-11  ConnectionError:   DB connection used without null-check
  BUG-12  RecursionError:    missing base case in recursive resolver

Each bug is in a separate, realistic endpoint or utility function
so the CPG extractor produces distinct graph nodes for each.
"""
import fastapi
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session
import requests

app = FastAPI(title="Eidolon Demo — Multi-Error Service")


# ─────────────────────────────────────────────────────────────────────────────
# BUG-01 · NameError  (undefined_reference)
# Cause:   decode_jwt is called but was never imported. Python only raises this
#          at runtime when the endpoint is hit, not at import time.
# Fix:     Add `from auth_utils import decode_jwt` at the top.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/verify-token")
def verify_user_token(jwt_token: str, db_session: Session):
    decoded_payload = decode_jwt(jwt_token)           # BUG-01: NameError
    if not decoded_payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_record = db_session.query(User).filter(User.id == decoded_payload["sub"]).first()
    return user_record


# ─────────────────────────────────────────────────────────────────────────────
# BUG-02 · AttributeError  (wrong_type_or_missing_method)
# Cause:   SQLAlchemy's Query object has no .last() method; .first() exists.
# Fix:     Replace .last() with .first()
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/user-profile")
def get_user_profile(user_id: int, db_session: Session):
    user = db_session.query(User).filter(User.id == user_id).last()   # BUG-02: AttributeError
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "email": user.email}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-03 · TypeError  (type_contract_violation)
# Cause:   send_welcome_email expects a str email but receives an int user_id.
# Fix:     Pass `user.email` (str) instead of `user.id` (int).
# ─────────────────────────────────────────────────────────────────────────────
def send_welcome_email(email: str) -> bool:
    """Send a welcome email. Expects a string email address."""
    if not isinstance(email, str):
        raise TypeError(f"email must be str, got {type(email).__name__}")
    return True

@app.post("/register")
def register_user(email: str, db_session: Session):
    new_user = User(email=email)
    db_session.add(new_user)
    db_session.flush()
    send_welcome_email(new_user.id)    # BUG-03: TypeError — should be new_user.email
    db_session.commit()
    return {"status": "registered", "id": new_user.id}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-04 · KeyError  (missing_key)
# Cause:   Config dict access with [] instead of .get() — crashes if key absent.
# Fix:     Use config.get("max_retries", 3) instead of config["max_retries"].
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_CONFIG = {"timeout_seconds": 30, "base_url": "https://api.internal"}

@app.get("/service-config")
def get_service_config():
    max_retries = SERVICE_CONFIG["max_retries"]   # BUG-04: KeyError — key missing
    timeout     = SERVICE_CONFIG["timeout_seconds"]
    return {"max_retries": max_retries, "timeout": timeout}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-05 · IndexError  (bounds_violation)
# Cause:   Always takes index [0] without checking if the list is non-empty.
# Fix:     Add a length check before indexing.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/latest-event")
def get_latest_event(db_session: Session):
    events = db_session.query(Event).order_by(Event.created_at.desc()).limit(5).all()
    latest = events[0]            # BUG-05: IndexError when events list is empty
    return {"event_id": latest.id, "type": latest.event_type}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-06 · ValueError  (invalid_value)
# Cause:   int() called on a raw query param that may contain non-numeric chars.
# Fix:     Add try/except ValueError or validate before conversion.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/invoice")
def get_invoice(invoice_id: str):
    numeric_id = int(invoice_id)   # BUG-06: ValueError if invoice_id = "INV-042"
    return {"invoice_id": numeric_id, "status": "pending"}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-07 · ImportError  (missing_dependency)
# Cause:   cryptography_utils is imported inside the function but the module
#          does not exist in the project.
# Fix:     Either install the package or use the stdlib `hmac` module instead.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/encrypt-payload")
def encrypt_payload(data: str):
    import cryptography_utils          # BUG-07: ImportError — module not installed
    encrypted = cryptography_utils.encrypt(data)
    return {"encrypted": encrypted}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-08 · AssertionError  (assertion_failure)
# Cause:   Precondition assertion fires when amount <= 0 is passed.
# Fix:     Replace assert with a proper HTTP 422 validation error.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/transfer-funds")
def transfer_funds(from_account: str, to_account: str, amount: float):
    assert amount > 0, "Transfer amount must be positive"   # BUG-08: AssertionError
    # ... process transfer
    return {"status": "transferred", "amount": amount}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-09 · PermissionError  (auth_failure)
# Cause:   Admin action performed without verifying the caller's role.
# Fix:     Add role check before executing the delete operation.
# ─────────────────────────────────────────────────────────────────────────────
def check_admin_role(token: str) -> None:
    """Raises PermissionError if caller is not an admin."""
    if not token.startswith("admin_"):
        raise PermissionError("Admin role required for this operation")

@app.delete("/admin/user/{user_id}")
def delete_user(user_id: int, db_session: Session, auth_token: str = ""):
    # BUG-09: check_admin_role is defined but never called here
    user = db_session.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db_session.delete(user)
    db_session.commit()
    return {"status": "deleted"}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-10 · TimeoutError  (timeout)
# Cause:   External HTTP call with no timeout specified. Hangs forever under load.
# Fix:     Add timeout=5 to the requests.get() call.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health-check")
def external_health_check():
    response = requests.get("https://api.internal/ping")   # BUG-10: no timeout
    return {"upstream_status": response.status_code}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-11 · ConnectionError  (network)
# Cause:   db_session used without checking if the connection is still alive.
# Fix:     Add connection validity check / reconnect logic before query.
# ─────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    """Returns a DB session — may return None if pool is exhausted."""
    return None   # Simulates exhausted connection pool

@app.get("/reports/summary")
def reports_summary():
    conn = get_db_connection()
    result = conn.execute("SELECT COUNT(*) FROM events")   # BUG-11: ConnectionError (NoneType)
    return {"event_count": result.fetchone()[0]}


# ─────────────────────────────────────────────────────────────────────────────
# BUG-12 · RecursionError  (infinite_recursion)
# Cause:   resolve_category calls itself with the same input when parent is None,
#          missing a base-case termination condition.
# Fix:     Add `if parent_id is None: return "root"` as the base case.
# ─────────────────────────────────────────────────────────────────────────────
def resolve_category(category_id: int, db_session: Session) -> str:
    """Walk up category tree to find root label."""
    row = db_session.query(Category).filter(Category.id == category_id).first()
    if row is None:
        return resolve_category(category_id, db_session)   # BUG-12: RecursionError — should return "root"
    if row.parent_id is None:
        return row.label
    return resolve_category(row.parent_id, db_session)

@app.get("/category/{category_id}")
def get_category_label(category_id: int, db_session: Session):
    label = resolve_category(category_id, db_session)
    return {"category_id": category_id, "label": label}
