"""SSL utility — default SSL context with Windows system cert store support."""

import logging
import os
import ssl

from src.config import settings

logger = logging.getLogger(__name__)

_SSL_ENVVARS_SET = False


def _apply_env_overrides():
    """Set env vars that third-party SDKs (zai, llama-parse, etc.) respect."""
    global _SSL_ENVVARS_SET
    if _SSL_ENVVARS_SET:
        return
    if not settings.ssl_verify:
        os.environ.setdefault("CURL_CA_BUNDLE", "")
        os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    elif settings.ssl_cert_bundle:
        bundle = os.path.abspath(settings.ssl_cert_bundle)
        if os.path.isfile(bundle):
            os.environ.setdefault("SSL_CERT_FILE", bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
            os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    _SSL_ENVVARS_SET = True


def create_ssl_context() -> ssl.SSLContext:
    """Create a default SSL context using Windows system certificate store.

    On Windows, ``ssl.create_default_context()`` uses the system CA store
    when ``pip-system-certs`` is installed (patches certifi to include
    Windows trust store certificates).
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED

    if settings.ssl_cert_bundle:
        bundle = os.path.abspath(settings.ssl_cert_bundle)
        if os.path.isfile(bundle):
            ctx.load_verify_locations(bundle)
        else:
            logger.warning("SSL_CERT_BUNDLE file not found: %s", bundle)

    return ctx


def get_verify_param() -> bool | ssl.SSLContext:
    """Return the ``verify`` value for ``httpx`` based on settings.

    Returns:
        - ``ssl.SSLContext`` — default context with Windows system cert store.
        - ``False``          — SSL verification disabled.
    """
    _apply_env_overrides()
    if not settings.ssl_verify:
        logger.warning("SSL verification is DISABLED — do not use in production")
        return False
    try:
        return create_ssl_context()
    except Exception as exc:
        logger.warning("Failed to create SSL context, using default: %s", exc)
        return True


def get_httpx_client():
    """Build an ``httpx.Client`` with default SSL context."""
    import httpx
    verify = get_verify_param()
    return httpx.Client(verify=verify)
