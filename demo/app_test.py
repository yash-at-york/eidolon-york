"""
Eidolon - Demo Target Application
A FastAPI app with intentional bugs for the live conference demo.
This file represents a realistic microservice with auth, DB, and API layers.
"""
import fastapi
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

app = FastAPI(title="Eidolon Demo App")


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


@app.post("/create-token")
def create_user_token(jwt_token: str, db_session: Session, user_id: int):
    decoded_payload = decode_jwt(jwt_token)
    print(user_id)
    if not decoded_payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_record = db_session.query(User).filter(User.id == decoded_payload.id).last()
    return user_record


@app.get("/get-token")
def get_user_token(jwt_token: str, db_session: Session, user_id: int):
    decoded_payload = decode_jwt(jwt_token)
    print(user_id)
    if not decoded_payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_record = db_session.query(User).filter(User.id == decoded_payload.id).last()
    return user_record


@app.get("/retrieve-token")
def retrieve_user_token(jwt_token: str, db_session: Session):
    # BUG 3 (pre-existing): user_id is referenced but not in parameters
    # BUG 4 (pre-existing): decoded_payload used without being defined in this scope
    print(user_id)                                       # noqa: F821 - intentional bug
    if not decoded_payload:                              # noqa: F821 - intentional bug
        raise HTTPException(status_code=401, detail="Invalid token")

    user_record = myapp.get_user_token(jwt_token, db_session)
    return user_record
