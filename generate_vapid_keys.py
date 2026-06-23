import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


private_key = ec.generate_private_key(ec.SECP256R1())
public_key = private_key.public_key()

public_numbers = public_key.public_numbers()

raw_public = (
    b"\x04"
    + public_numbers.x.to_bytes(32, "big")
    + public_numbers.y.to_bytes(32, "big")
)

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode("utf-8")

with open("vapid_private_key.pem", "w", encoding="utf-8") as f:
    f.write(private_pem)

print("\nAdd these to your .env file:\n")
print("VAPID_PUBLIC_KEY=" + b64url(raw_public))
print("VAPID_PRIVATE_KEY_FILE=vapid_private_key.pem")
print("VAPID_CLAIM_EMAIL=mailto:srivathsan659@gmail.com")
print("\nPrivate key saved to: vapid_private_key.pem")
print("Keep vapid_private_key.pem private. Do not upload it publicly.\n")