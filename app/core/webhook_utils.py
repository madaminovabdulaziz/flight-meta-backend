# app/core/webhook_utils.py
import hmac
import hashlib
import json
from typing import Dict, Any


def generate_webhook_signature(payload: Dict[Any, Any], secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.
    
    Use this function when you need to:
    1. Test your webhook endpoint
    2. Verify partner integration
    
    Args:
        payload: Dictionary to send as webhook
        secret: Partner's webhook secret
    
    Returns:
        Signature in format "sha256=<hash>"
    
    Example:
        >>> payload = {"click_id": 123, "commission": 15.50}
        >>> secret = "my_secret_key"
        >>> sig = generate_webhook_signature(payload, secret)
        >>> # Send in header: X-Webhook-Signature: {sig}
    """
    # Convert payload to JSON bytes (same as FastAPI will receive)
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    
    # Generate HMAC
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    
    return f"sha256={signature}"


# Example usage / testing script
if __name__ == "__main__":
    # Test webhook signature generation
    test_payload = {
        "partner_name": "Amadeus",
        "click_id": 12345,
        "partner_booking_ref": "ABC123",
        "commission_amount": 15.50,
        "commission_currency": "USD"
    }
    
    secret = "test_webhook_secret_key"
    
    signature = generate_webhook_signature(test_payload, secret)
    
    print("Test Webhook Request:")
    print("-" * 50)
    print(f"URL: POST /api/v1/webhooks/partner-conversion")
    print(f"Header: X-Webhook-Signature: {signature}")
    print(f"Body: {json.dumps(test_payload, indent=2)}")
    print("-" * 50)
    
    # You can use this output to test with curl:
    print("\nCURL command:")
    print(f"""
curl -X POST http://localhost:8000/api/v1/webhooks/partner-conversion \\
  -H "Content-Type: application/json" \\
  -H "X-Webhook-Signature: {signature}" \\
  -d '{json.dumps(test_payload)}'
""")