# app/api/v1/endpoints/auth.py
"""
Authentication endpoints - FIXED
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import timedelta
import logging
from app.api.v1.dependencies import get_current_user
from app.db.database import get_db
from app.models.models import User
from schemas.users import UserCreate, User as UserSchema
from schemas.token import Token
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=UserSchema, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate, 
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user.
    
    Returns:
        User object with id and email (password is hidden)
    """
    try:
        # Check if user already exists
        result = await db.execute(
            select(User).where(User.email == user_in.email.lower())
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists."
            )
        
        # Validate password strength
        if len(user_in.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long."
            )
        
        if len(user_in.password) > 128:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password cannot be longer than 128 characters."
            )
        
        # Hash password
        hashed_password = get_password_hash(user_in.password)
        
        # Create user
        new_user = User(
            email=user_in.email.lower(),  # Store emails in lowercase
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        logger.info(f"New user registered: {new_user.email}")
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration."
        )


@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return a JWT token.
    
    OAuth2 compatible - use 'username' field for email.
    
    Returns:
        JWT access token
    """
    try:
        # Find user by email (username field in OAuth2)
        result = await db.execute(
            select(User).where(User.email == form_data.username.lower())
        )
        user = result.scalar_one_or_none()
        
        # Check if user exists and password is correct
        if not user or not verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Failed login attempt for: {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            subject=user.id,
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user.email}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login."
        )


@router.post("/test-token", response_model=UserSchema)
async def test_token(
    current_user: User = Depends(get_current_user)
):
    """
    Test if the JWT token is valid.
    Returns current user info.
    """
    return current_user


@router.get("/me", response_model=UserSchema)
async def read_users_me(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information.
    """
    return current_user


# Import dependency
from app.api.v1.dependencies import get_current_user


# from fastapi import APIRouter, Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordRequestForm
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.future import select
# from datetime import timedelta
# from app.db.database import get_db
# from app.models.models import User
# from schemas.users import UserCreate, User as UserSchema
# from schemas.token import Token
# from app.core.security import get_password_hash, verify_password, create_access_token
# from app.core.config import settings
# router = APIRouter()

# @router.post("/register", response_model=UserSchema)
# async def register_user(
#     user_in: UserCreate, 
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Create a new user.
#     """
#     # Check if user already exists
#     result = await db.execute(select(User).where(User.email == user_in.email))
#     existing_user = result.scalar_one_or_none()
#     if existing_user:
#         raise HTTPException(
#             status_code=400,
#             detail="The user with this email already exists in the system.",
#         )

#     hashed_password = get_password_hash(user_in.password)
#     new_user = User(email=user_in.email, hashed_password=hashed_password)
#     db.add(new_user)
#     await db.commit()
#     await db.refresh(new_user)
#     return new_user

# @router.post("/login", response_model=Token)
# async def login_for_access_token(
#     db: AsyncSession = Depends(get_db), 
#     form_data: OAuth2PasswordRequestForm = Depends()
# ):
#     """
#     Authenticate user and return a JWT token.
#     """
#     result = await db.execute(select(User).where(User.email == form_data.username))
#     user = result.scalar_one_or_none()
    
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect email or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         subject=user.id, expires_delta=access_token_expires
#     )
#     return {"access_token": access_token, "token_type": "bearer"}