# services/payment_service.py
import stripe
import httpx
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.models.models import Payment, Booking, User
from app.models.models import PaymentStatus, PaymentMethod, BookingStatus
from schemas.payment import PaymentCreate, PaymentResponse

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaymentService:
    """Unified payment service for Stripe, Click, and Payme"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_payment(
        self,
        user: User,
        payment_data: PaymentCreate
    ) -> PaymentResponse:
        """
        Create a new payment based on the selected method
        """
        # Verify booking exists and belongs to user
        result = await self.db.execute(
            select(Booking).where(
                Booking.id == payment_data.booking_id,
                Booking.user_id == user.id
            )
        )
        booking = result.scalar_one_or_none()

        if not booking:
            raise ValueError("Booking not found or does not belong to user")

        if booking.status not in [BookingStatus.PENDING_PAYMENT, BookingStatus.CONFIRMED]:
            raise ValueError(f"Booking cannot be paid. Current status: {booking.status}")

        # Verify amount matches booking
        if abs(payment_data.amount - booking.total_price) > 0.01:
            raise ValueError(
                f"Payment amount {payment_data.amount} does not match booking total {booking.total_price}"
            )

        # Generate unique payment ID
        payment_id = self._generate_payment_id()

        # Create payment record
        payment = Payment(
            payment_id=payment_id,
            user_id=user.id,
            booking_id=booking.id,
            amount=payment_data.amount,
            currency=payment_data.currency,
            method=payment_data.method,  # Already validated by Pydantic Enum
            status=PaymentStatus.PENDING,
            provider=self._get_provider_name(payment_data.method)
        )

        self.db.add(payment)
        await self.db.flush()

        # Process payment based on method
        try:
            if payment_data.method == PaymentMethod.CARD:
                result = await self._process_stripe_payment(payment, booking)
            elif payment_data.method == PaymentMethod.CLICK:
                result = await self._process_click_payment(payment, booking, payment_data.return_url)
            elif payment_data.method == PaymentMethod.PAYME:
                result = await self._process_payme_payment(payment, booking, payment_data.return_url)
            else:
                raise ValueError(f"Unsupported payment method: {payment_data.method}")

            await self.db.commit()
            await self.db.refresh(payment)

            return PaymentResponse(
                id=payment.id,
                payment_id=payment.payment_id,
                booking_id=payment.booking_id,
                user_id=payment.user_id,
                amount=payment.amount,
                currency=payment.currency,
                method=payment.method,
                status=payment.status,
                provider=payment.provider,
                provider_payment_id=payment.provider_payment_id,
                payment_url=result.get("payment_url"),
                paid_at=payment.paid_at,
                created_at=payment.created_at
            )

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Payment creation failed: {e}")
            raise

    def _generate_payment_id(self) -> str:
        """Generate unique payment ID"""
        import random
        import string
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"PAY-{timestamp}-{random_suffix}"

    def _get_provider_name(self, method: PaymentMethod) -> str:
        """Get provider name from payment method"""
        mapping = {
            PaymentMethod.CARD: "stripe",
            PaymentMethod.CLICK: "click",
            PaymentMethod.PAYME: "payme",
            PaymentMethod.CASH: "cash"
        }
        return mapping.get(method, "unknown")

    # === STRIPE INTEGRATION ===
    async def _process_stripe_payment(
        self,
        payment: Payment,
        booking: Booking
    ) -> Dict[str, Any]:
        """Process payment via Stripe"""
        try:
            intent = stripe.PaymentIntent.create(
                amount=int(payment.amount * 100),
                currency=payment.currency.lower(),
                metadata={
                    "payment_id": payment.payment_id,
                    "booking_id": str(booking.id),
                    "pnr_reference": booking.pnr_reference or ""
                },
                automatic_payment_methods={"enabled": True}
            )

            payment.provider_payment_id = intent.id
            payment.provider_response = intent.to_dict()
            payment.status = PaymentStatus.PROCESSING

            logger.info(f"Stripe payment intent created: {intent.id}")

            return {
                "client_secret": intent.client_secret,
                "payment_intent_id": intent.id
            }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {e}")
            payment.status = PaymentStatus.FAILED
            raise ValueError(f"Stripe payment failed: {str(e)}")

    async def handle_stripe_webhook(self, event: Dict[str, Any]) -> None:
        """Handle Stripe webhook events"""
        event_type = event["type"]

        if event_type == "payment_intent.succeeded":
            await self._handle_stripe_success(event["data"]["object"])
        elif event_type == "payment_intent.payment_failed":
            await self._handle_stripe_failure(event["data"]["object"])

    async def _handle_stripe_success(self, payment_intent: Dict[str, Any]) -> None:
        """Handle successful Stripe payment"""
        result = await self.db.execute(
            select(Payment).where(Payment.provider_payment_id == payment_intent["id"])
        )
        payment = result.scalar_one_or_none()

        if payment:
            payment.status = PaymentStatus.COMPLETED
            payment.paid_at = datetime.now()

            result = await self.db.execute(
                select(Booking).where(Booking.id == payment.booking_id)
            )
            booking = result.scalar_one_or_none()
            if booking:
                booking.status = BookingStatus.TICKETED

            await self.db.commit()
            logger.info(f"Payment {payment.payment_id} completed successfully")

    async def _handle_stripe_failure(self, payment_intent: Dict[str, Any]) -> None:
        """Handle failed Stripe payment"""
        result = await self.db.execute(
            select(Payment).where(Payment.provider_payment_id == payment_intent["id"])
        )
        payment = result.scalar_one_or_none()

        if payment:
            payment.status = PaymentStatus.FAILED
            await self.db.commit()
            logger.warning(f"Payment {payment.payment_id} failed")

    # === CLICK + PAYME integration remain unchanged except enums ===
    # Replace string checks with PaymentMethod enums
