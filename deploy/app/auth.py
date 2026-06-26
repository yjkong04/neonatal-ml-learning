"""HTTP Basic Auth — the fallback layer behind Cloudflare Access.

Single user, credentials from env vars (set as Fly secrets in production).
Always on for every route except /healthz. This exists so the raw
*.fly.dev origin is never an open bypass if Access is ever misconfigured.
"""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    expected_user = os.environ.get("DEPLOY_USER", "")
    expected_password = os.environ.get("DEPLOY_PASSWORD", "")

    # Constant-time comparison — avoids leaking match length via response timing.
    user_ok = secrets.compare_digest(credentials.username, expected_user)
    password_ok = secrets.compare_digest(credentials.password, expected_password)

    if not (user_ok and password_ok and expected_user and expected_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
