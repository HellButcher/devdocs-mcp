"""Utility functions for HTTP requests with retry logic."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    backoff: float = 1.0,
    operation_name: str = "operation",
) -> T:
    """Execute a function with retry logic and exponential backoff.
    
    Args:
        func: Function to execute (should raise exception on failure)
        max_retries: Maximum number of retry attempts
        backoff: Initial backoff delay in seconds (doubles each retry)
        operation_name: Description of operation for logging
        
    Returns:
        Result of successful function execution
        
    Raises:
        Exception: The last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.debug(
                "%s (attempt %d/%d)",
                operation_name, attempt + 1, max_retries
            )
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = backoff * (2 ** attempt)
                logger.warning(
                    "%s failed: %s. Retrying in %.1fs...",
                    operation_name, e, delay
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name, max_retries, e
                )
    
    raise last_error  # type: ignore


def http_get_with_retry(
    url: str,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff: float = 1.0,
) -> httpx.Response:
    """Make HTTP GET request with retry logic.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff: Initial backoff delay in seconds (doubles each retry)
        
    Returns:
        HTTP response
        
    Raises:
        httpx.HTTPError: If all retries fail
    """
    def request() -> httpx.Response:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp
    
    return retry_with_backoff(
        request,
        max_retries=max_retries,
        backoff=backoff,
        operation_name=f"HTTP GET {url}",
    )


def http_download_with_retry(
    url: str,
    timeout: float = 120.0,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> bytes:
    """Download file with retry logic.
    
    Args:
        url: URL to download
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        backoff: Initial backoff delay in seconds
        
    Returns:
        Response content bytes
        
    Raises:
        httpx.HTTPError: If download fails after retries
    """
    def download() -> bytes:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    
    return retry_with_backoff(
        download,
        max_retries=max_retries,
        backoff=backoff,
        operation_name=f"Download {url}",
    )
