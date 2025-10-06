# app/api/v1/endpoints/clicks.py
from fastapi import APIRouter, Depends, Request, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import uuid

from app.db.database import get_db
from app.models.models import User, Click, Partner
from app.api.v1.dependencies import get_current_user, get_current_user_optional
from schemas.clicks import ClickCreate, ClickResponse, ClickStats, ClickHistoryResponse

router = APIRouter()


def hash_ip(ip: str) -> str:
    """Hash IP address for privacy"""
    return hashlib.sha256(ip.encode()).hexdigest()


def get_or_create_session_id(request: Request, response: Response) -> str:
    """Get session ID from cookie or create new one"""
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        # Set cookie for 30 days
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=30 * 24 * 60 * 60,  # 30 days
            httponly=True,
            secure=True,  # FIXED: Only send over HTTPS
            samesite="lax"
        )
    
    return session_id


@router.post("/", response_model=ClickResponse, status_code=201)
async def track_click(
    click_data: ClickCreate,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Track when user clicks through to partner site.
    
    This is the CRITICAL revenue tracking endpoint.
    Call this immediately before redirecting user to partner.
    
    Returns:
    - click_id: For tracking conversions
    - redirect_url: Where to send the user
    - session_id: For tracking anonymous users
    """
    # Get or create session ID
    session_id = get_or_create_session_id(request, response)
    
    # Validate partner exists
    partner_result = await db.execute(
        select(Partner).where(
            and_(
                Partner.name == click_data.partner_name,
                Partner.is_active == True
            )
        )
    )
    partner = partner_result.scalar_one_or_none()
    
    if not partner:
        raise HTTPException(
            status_code=400,
            detail=f"Partner '{click_data.partner_name}' not found or inactive"
        )
    
    # FIXED: Calculate expected commission separately from earned
    commission_expected = None
    if partner.commission_rate:
        if partner.commission_model == "CPA":
            commission_expected = partner.commission_rate
        elif partner.commission_model == "Revenue Share":
            commission_expected = click_data.price * (partner.commission_rate / 100)
    
    # Create click record
    click = Click(
        user_id=current_user.id if current_user else None,
        session_id=session_id,
        origin=click_data.origin.upper(),
        destination=click_data.destination.upper(),
        departure_date=click_data.departure_date,
        return_date=click_data.return_date,
        price=click_data.price,
        currency=click_data.currency.upper(),
        partner_id=partner.id,
        partner_name=partner.name,  # Denormalized for fast analytics queries
        partner_deeplink=click_data.deeplink,
        flight_offer_snapshot=click_data.flight_offer_snapshot,
        # Store expected separately - actual commission updated via webhook
        commission_earned=None,  # FIXED: Set to None, updated on conversion
        ip_hash=hash_ip(request.client.host) if request.client else None,
        user_agent=request.headers.get("user-agent"),
        referrer=request.headers.get("referer")
    )
    
    db.add(click)
    await db.commit()
    await db.refresh(click)
    
    return ClickResponse(
        click_id=click.id,
        redirect_url=click.partner_deeplink,
        session_id=session_id,
        partner_name=click.partner_name
    )


@router.get("/history", response_model=list[ClickHistoryResponse])
async def get_click_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 20
):
    """
    Get user's click history.
    Shows which flights they've clicked through to book.
    """
    result = await db.execute(
        select(Click)
        .where(Click.user_id == current_user.id)
        .order_by(Click.clicked_at.desc())
        .offset(skip)
        .limit(limit)
    )
    clicks = result.scalars().all()
    
    return [
        ClickHistoryResponse(
            id=click.id,
            origin=click.origin,
            destination=click.destination,
            departure_date=click.departure_date,
            return_date=click.return_date,
            price=click.price,
            currency=click.currency,
            partner_name=click.partner_name,
            converted=click.converted,
            clicked_at=click.clicked_at
        )
        for click in clicks
    ]


@router.get("/stats", response_model=ClickStats)
async def get_click_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user's click statistics.
    Useful for showing "You've saved X by using our platform" messaging.
    """
    # Total clicks
    total_result = await db.execute(
        select(func.count(Click.id))
        .where(Click.user_id == current_user.id)
    )
    total_clicks = total_result.scalar()
    
    # Total amount clicked
    amount_result = await db.execute(
        select(func.sum(Click.price))
        .where(Click.user_id == current_user.id)
    )
    total_amount = amount_result.scalar() or 0.0
    
    # Converted bookings
    converted_result = await db.execute(
        select(func.count(Click.id))
        .where(
            and_(
                Click.user_id == current_user.id,
                Click.converted == True
            )
        )
    )
    converted_count = converted_result.scalar()
    
    # Last 30 days clicks
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_result = await db.execute(
        select(func.count(Click.id))
        .where(
            and_(
                Click.user_id == current_user.id,
                Click.clicked_at >= thirty_days_ago
            )
        )
    )
    recent_clicks = recent_result.scalar()
    
    return ClickStats(
        total_clicks=total_clicks,
        total_amount_clicked=total_amount,
        converted_bookings=converted_count,
        recent_clicks_30d=recent_clicks
    )


@router.patch("/{click_id}/converted", status_code=200)
async def mark_click_converted(
    click_id: int,
    commission_amount: Optional[float] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    DEPRECATED: Use /webhooks/partner-conversion endpoint instead.
    
    This endpoint is kept for backward compatibility but should not be used.
    The webhook endpoint has proper security (signature verification).
    """
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /webhooks/partner-conversion instead."
    )


@router.get("/analytics/top-routes")
async def get_top_routes(
    db: AsyncSession = Depends(get_db),
    limit: int = 10,
    days: int = 30
):
    """
    ADMIN/ANALYTICS: Get most clicked routes.
    Useful for understanding user behavior and popular destinations.
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    result = await db.execute(
        select(
            Click.origin,
            Click.destination,
            func.count(Click.id).label("click_count"),
            func.avg(Click.price).label("avg_price")
        )
        .where(Click.clicked_at >= cutoff_date)
        .group_by(Click.origin, Click.destination)
        .order_by(func.count(Click.id).desc())
        .limit(limit)
    )
    
    routes = result.all()
    
    return [
        {
            "route": f"{route.origin}-{route.destination}",
            "click_count": route.click_count,
            "avg_price": round(route.avg_price, 2)
        }
        for route in routes
    ]


@router.get("/analytics/conversion-rate")
async def get_conversion_rate(
    db: AsyncSession = Depends(get_db),
    days: int = 30
):
    """
    ADMIN/ANALYTICS: Calculate conversion rate.
    Shows what % of clicks result in actual bookings.
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Total clicks
    total_result = await db.execute(
        select(func.count(Click.id))
        .where(Click.clicked_at >= cutoff_date)
    )
    total = total_result.scalar()
    
    # Converted clicks
    converted_result = await db.execute(
        select(func.count(Click.id))
        .where(
            and_(
                Click.clicked_at >= cutoff_date,
                Click.converted == True
            )
        )
    )
    converted = converted_result.scalar()
    
    conversion_rate = (converted / total * 100) if total > 0 else 0
    
    return {
        "period_days": days,
        "total_clicks": total,
        "converted_clicks": converted,
        "conversion_rate_percent": round(conversion_rate, 2)
    }