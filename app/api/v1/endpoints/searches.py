from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.db.database import get_db
from app.models.models import User, SavedSearch

from schemas.saved_search import SavedSearchCreate, SavedSearchUpdate, SavedSearch as SavedSearchSchema
from app.api.v1.dependencies import get_current_user

router = APIRouter()


# ----------------------------
# CREATE
# ----------------------------
@router.post("/", response_model=SavedSearchSchema)
async def create_saved_search(
    search_in: SavedSearchCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Save a new flight search for the current user.
    """
    new_search = SavedSearch(
        **search_in.model_dump(),
        user_id=current_user.id
    )
    db.add(new_search)
    await db.commit()
    await db.refresh(new_search)
    return new_search


# ----------------------------
# READ ALL (with pagination)
# ----------------------------
@router.get("/", response_model=List[SavedSearchSchema])
async def read_saved_searches(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve all saved searches for the current user (with pagination).
    """
    result = await db.execute(
        select(SavedSearch)
        .where(SavedSearch.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    searches = result.scalars().all()
    return searches


# ----------------------------
# READ ONE
# ----------------------------
@router.get("/{search_id}", response_model=SavedSearchSchema)
async def read_saved_search(
    search_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve a single saved search by ID.
    """
    result = await db.execute(
        select(SavedSearch).where(
            SavedSearch.id == search_id,
            SavedSearch.user_id == current_user.id
        )
    )
    search = result.scalars().first()
    if not search:
        raise HTTPException(status_code=404, detail="Saved search not found")
    return search


# ----------------------------
# UPDATE
# ----------------------------
@router.put("/{search_id}", response_model=SavedSearchSchema)
async def update_saved_search(
    search_id: int,
    search_in: SavedSearchUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update a saved search by ID.
    """
    result = await db.execute(
        select(SavedSearch).where(
            SavedSearch.id == search_id,
            SavedSearch.user_id == current_user.id
        )
    )
    search = result.scalars().first()
    if not search:
        raise HTTPException(status_code=404, detail="Saved search not found")

    # Update fields dynamically
    for field, value in search_in.model_dump(exclude_unset=True).items():
        setattr(search, field, value)

    db.add(search)
    await db.commit()
    await db.refresh(search)
    return search


# ----------------------------
# DELETE
# ----------------------------
@router.delete("/{search_id}", response_model=dict)
async def delete_saved_search(
    search_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Delete a saved search by ID if it belongs to the current user.
    """
    result = await db.execute(
        select(SavedSearch).where(
            SavedSearch.id == search_id,
            SavedSearch.user_id == current_user.id
        )
    )
    search = result.scalars().first()
    if not search:
        raise HTTPException(status_code=404, detail="Saved search not found")

    await db.delete(search)
    await db.commit()
    return {"detail": "Saved search deleted successfully"}
