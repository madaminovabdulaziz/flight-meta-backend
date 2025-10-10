# app/core/security.py - FINAL FIX
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import jwt
from typing import Any, Union
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Bcrypt context - simple and clean
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"


def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject), "iat": datetime.now(timezone.utc)}
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password - handles 72 byte limit"""
    try:
        # Ensure password is a string and truncate to 72 bytes if needed
        password_str = str(plain_password)
        password_bytes = password_str.encode('utf-8')
        
        # If > 72 bytes, just truncate (bcrypt ignores anything after 72 anyway)
        if len(password_bytes) > 72:
            password_str = password_bytes[:72].decode('utf-8', errors='ignore')
        
        return pwd_context.verify(password_str, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash password - handles 72 byte limit"""
    try:
        # Ensure password is a string and truncate to 72 bytes if needed
        password_str = str(password)
        password_bytes = password_str.encode('utf-8')
        
        # If > 72 bytes, just truncate (bcrypt ignores anything after 72 anyway)
        if len(password_bytes) > 72:
            password_str = password_bytes[:72].decode('utf-8', errors='ignore')
        
        return pwd_context.hash(password_str)
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        raise

# from datetime import datetime, timedelta, timezone
# from passlib.context import CryptContext
# from jose import jwt
# from typing import Any, Union

# from app.core.config import settings

# pwd_context = CryptContext(
#     schemes=["bcrypt_sha256", "bcrypt"],  # verify old hashes too
#     deprecated="auto",
# )



# ALGORITHM = "HS256"

# def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None) -> str:
#     """
#     Creates a new JWT access token.
#     """
#     if expires_delta:
#         expire = datetime.now(timezone.utc) + expires_delta
#     else:
#         expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
#     to_encode = {"exp": expire, "sub": str(subject)}
#     encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """
#     Verifies a plain password against its hashed version.
#     """
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     """
#     Hashes a plain password.
#     """
#     return pwd_context.hash(password)

