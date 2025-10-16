# app/api/v1/dependencies.py
"""
FastAPI dependencies - FIXED
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from jose import JWTError, jwt
from typing import Optional
import logging

from app.db.database import get_db
from app.models.models import User
from app.core.config import settings

logger = logging.getLogger(__name__)

# OAuth2 scheme - points to login endpoint
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False  # Makes it optional
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Raises:
        HTTPException: 401 if token is invalid or user not found
    
    Returns:
        User object
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Token missing 'sub' claim")
            raise credentials_exception
        
        # Convert to int
        try:
            user_id = int(user_id)
        except ValueError:
            logger.error(f"Invalid user_id in token: {user_id}")
            raise credentials_exception
        
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise credentials_exception
    
    # Fetch user from database
    try:
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            logger.warning(f"User {user_id} not found in database")
            raise credentials_exception
        
        return user
        
    except Exception as e:
        logger.error(f"Database error fetching user: {e}")
        raise credentials_exception


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    
    Used for endpoints that work for both logged-in and anonymous users.
    
    Returns:
        User object if authenticated, None otherwise
    """
    if not token:
        return None
    
    try:
        # Try to decode token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        # Convert to int
        try:
            user_id = int(user_id)
        except ValueError:
            return None
        
        # Fetch user
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        return user
        
    except JWTError:
        return None
    except Exception as e:
        logger.error(f"Error in get_current_user_optional: {e}")
        return None


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (for future use if you add is_active field).
    
    For now, just returns current_user.
    """
    # TODO: Add is_active check if you add that field to User model
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
