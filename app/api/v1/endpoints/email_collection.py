from fastapi import APIRouter, HTTPException
from aiogram import Bot
import re
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Replace with your actual bot token and chat ID
BOT_TOKEN = "8003940524:AAF_YoHbPFGuIXDLz_2hTUax4XkM_revkoo"
CHAT_ID = "5069131343"
bot = Bot(token=BOT_TOKEN)

# Simple email regex validator
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"


@router.post("/post-email")
async def post_email(email: str):
    # Validate email format
    if not re.match(EMAIL_REGEX, email):
        logger.warning(f"Invalid email attempt: {email}")
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Send message to Telegram bot
    try:
        await bot.send_message(CHAT_ID, f"📧 New email submitted: {email}")
        logger.info(f"Email sent to bot: {email}")
        return {"status": "success", "email": email}
    except Exception as e:
        logger.error(f"Failed to send email to bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message to bot")
