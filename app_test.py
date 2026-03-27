import fastapi
from fastapi import FastAPI, HTTPException
from sqlalchemy.orm import Session

app = FastAPI()

@app.post("/verify-token")
def verify_user_token(jwt_token: str, db_session: Session, user_id: int):
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
    print(user_id)
    if not decoded_payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_record = myapp.get_user_token(jwt_token, db_session)
    return user_record