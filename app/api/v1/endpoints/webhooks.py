# app/api/v1/endpoints/webhooks.py
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import datetime
import hmac
import hashlib
import json
from typing import Optional

from app.db.database import get_db
from app.models.models import Click, Partner
from schemas.webhooks import PartnerConversionWebhook
from app.core.config import settings

router = APIRouter()


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str
) -> bool:
    """
    Verify webhook signature using HMAC-SHA256.
    
    Args:
        payload: Raw request body as bytes
        signature: Signature from header (format: "sha256=<hash>")
        secret: Partner's webhook secret
    
    Returns:
        True if signature is valid
    """
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    provided_signature = signature.split("=")[1]
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_signature, provided_signature)


async def get_partner_by_name(
    partner_name: str,
    db: AsyncSession
) -> Partner:
    """Get partner and validate it exists"""
    result = await db.execute(
        select(Partner).where(Partner.name == partner_name)
    )
    partner = result.scalar_one_or_none()
    
    if not partner:
        raise HTTPException(status_code=404, detail=f"Partner '{partner_name}' not found")
    
    return partner


@router.post("/partner-conversion")
async def handle_partner_conversion(
    request: Request,
    webhook_data: PartnerConversionWebhook,
    x_webhook_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """
    SECURE WEBHOOK: Partner notifies us when a click converts to booking.
    
    Security measures:
    1. HMAC signature verification
    2. Partner validation
    3. Idempotency (don't double-count conversions)
    
    Expected flow:
    1. User clicks through from your site (Click record created)
    2. User completes booking on partner site
    3. Partner sends webhook to this endpoint
    4. We mark the click as converted and record commission
    """
    # Get partner
    partner = await get_partner_by_name(webhook_data.partner_name, db)
    
    # Verify webhook signature
    if partner.webhook_secret:
        body = await request.body()
        
        if not x_webhook_signature:
            raise HTTPException(
                status_code=401,
                detail="Missing webhook signature"
            )
        
        if not verify_webhook_signature(body, x_webhook_signature, partner.webhook_secret):
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
    else:
        # If no secret configured, at least log warning
        # In production, you should REQUIRE webhook secrets
        import logging
        logging.warning(f"Partner {partner.name} has no webhook secret configured!")
    
    # Find the click record
    result = await db.execute(
        select(Click).where(Click.id == webhook_data.click_id)
    )
    click = result.scalar_one_or_none()
    
    if not click:
        raise HTTPException(
            status_code=404,
            detail=f"Click {webhook_data.click_id} not found"
        )
    
    # Idempotency check - prevent double-counting
    if click.converted:
        return {
            "message": "Conversion already recorded",
            "click_id": click.id,
            "already_converted": True
        }
    
    # Validate the click belongs to this partner
    if click.partner_id != partner.id:
        raise HTTPException(
            status_code=400,
            detail=f"Click {click.id} does not belong to partner {partner.name}"
        )
    
    # Update click with conversion data
    click.converted = True
    click.conversion_tracked_at = datetime.utcnow()
    click.commission_earned = webhook_data.commission_amount
    
    # Store partner's booking reference for reconciliation
    if not click.flight_offer_snapshot:
        click.flight_offer_snapshot = {}
    
    click.flight_offer_snapshot["partner_booking_ref"] = webhook_data.partner_booking_ref
    click.flight_offer_snapshot["webhook_received_at"] = datetime.utcnow().isoformat()
    
    await db.commit()
    await db.refresh(click)
    
    return {
        "message": "Conversion recorded successfully",
        "click_id": click.id,
        "commission_earned": click.commission_earned,
        "converted_at": click.conversion_tracked_at
    }


@router.post("/test-webhook")
async def test_webhook_signature(
    request: Request,
    partner_name: str,
    x_webhook_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """
    TEST ENDPOINT: Verify webhook signature configuration.
    
    Use this to test your webhook setup with a partner.
    Send any JSON payload and we'll verify the signature.
    """
    partner = await get_partner_by_name(partner_name, db)
    
    if not partner.webhook_secret:
        return {
            "success": False,
            "message": f"Partner {partner_name} has no webhook secret configured"
        }
    
    body = await request.body()
    
    if not x_webhook_signature:
        return {
            "success": False,
            "message": "No signature provided in X-Webhook-Signature header"
        }
    
    is_valid = verify_webhook_signature(body, x_webhook_signature, partner.webhook_secret)
    
    if is_valid:
        return {
            "success": True,
            "message": "Webhook signature is valid!",
            "partner": partner_name
        }
    else:
        return {
            "success": False,
            "message": "Webhook signature is INVALID",
            "hint": "Check that you're using HMAC-SHA256 and the correct secret"
        }