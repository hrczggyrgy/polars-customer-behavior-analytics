"""Security, privacy, and deployment utilities for the dashboard.

This module implements security best practices for handling customer transaction data.
"""

import hashlib
import json
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ============================================================================
# File Upload Security
# ============================================================================

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'.parquet', '.csv'}
ALLOWED_MIME_TYPES = {
    'application/octet-stream',  # Parquet
    'text/csv',
    'application/csv',
    'text/plain',
}

DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
    '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl', '.sh', '.ps1',
}


def validate_upload_file(
    file_path: Path,
    max_size: int = MAX_UPLOAD_SIZE,
    allowed_extensions: set | None = None,
    allow_temp_files: bool = False,
) -> tuple[bool, str]:
    """
    Validate uploaded file for security.
    
    Returns: (is_valid, error_message)
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_EXTENSIONS

    # Check file exists
    if not file_path.exists():
        return False, "File does not exist"

    # Check file size
    file_size = file_path.stat().st_size
    if file_size > max_size:
        return False, f"File size {file_size:,} bytes exceeds maximum {max_size:,} bytes"

    if file_size == 0:
        return False, "File is empty"

    # Check extension
    ext = file_path.suffix.lower()
    if ext not in allowed_extensions:
        return False, f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}"

    # Check for dangerous extensions (double extensions)
    if any(danger in file_path.name.lower() for danger in DANGEROUS_EXTENSIONS):
        return False, "File name contains potentially dangerous extension"

    # Check for path traversal attempts in the file name itself
    if '..' in file_path.name:
        return False, "Invalid file name (path traversal attempt)"

    # Allow absolute paths for temp files if flag is set
    if file_path.is_absolute() and not allow_temp_files:
        # Check if it's in a system temp directory
        temp_dirs = {Path(tempfile.gettempdir()), Path('/tmp'), Path('/var/tmp')}
        try:
            resolved = file_path.resolve()
            if not any(str(resolved).startswith(str(d)) for d in temp_dirs):
                return False, "Absolute paths not allowed for uploads"
        except Exception:
            return False, "Invalid file path"

    return True, "OK"


def validate_parquet_file(file_path: Path) -> tuple[bool, str]:
    """
    Validate Parquet file structure and content.
    
    Checks for valid Parquet magic bytes and basic structure.
    """
    try:
        with file_path.open('rb') as f:
            # Check magic bytes
            header = f.read(4)
            if header != b'PAR1':
                return False, "Not a valid Parquet file (missing PAR1 header)"

            f.seek(-4, 2)
            footer = f.read(4)
            if footer != b'PAR1':
                return False, "Not a valid Parquet file (missing PAR1 footer)"

        return True, "OK"
    except Exception as e:
        return False, f"Parquet validation failed: {e}"


def validate_csv_file(file_path: Path) -> tuple[bool, str]:
    """
    Validate CSV file for basic safety.
    
    Checks for formula injection attempts in first few rows.
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            # Read first 1000 characters
            sample = f.read(1000)

            # Check for formula injection patterns
            dangerous_patterns = ['=', '+', '-', '@', '\t', '\r']
            lines = sample.split('\n')

            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                cells = line.split(',')
                for cell in cells:
                    cell = cell.strip().strip('"\'')
                    if cell and cell[0] in dangerous_patterns:
                        # This is a potential formula injection
                        # We'll escape it on export, but flag for review
                        return True, "WARNING: Potential formula injection detected in CSV"

        return True, "OK"
    except UnicodeDecodeError:
        return False, "CSV file is not valid UTF-8"
    except Exception as e:
        return False, f"CSV validation failed: {e}"


# ============================================================================
# Secure Temporary Workspace
# ============================================================================

class SecureWorkspace:
    """Manages a secure temporary workspace with automatic cleanup."""

    def __init__(
        self,
        prefix: str = "retail_analytics_",
        max_age_hours: int = 24,
        base_dir: Path | None = None,
    ):
        self.prefix = prefix
        self.max_age = timedelta(hours=max_age_hours)
        self.base_dir = base_dir or Path(tempfile.gettempdir())
        self.workspace: Path | None = None
        self.created_at: datetime | None = None

    def __enter__(self) -> Path:
        self.workspace = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.base_dir))
        self.created_at = datetime.now()
        # Set restrictive permissions
        os.chmod(self.workspace, 0o700)
        return self.workspace

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up workspace and all contents."""
        if self.workspace and self.workspace.exists():
            try:
                shutil.rmtree(self.workspace, ignore_errors=True)
            except Exception as e:
                warnings.warn(f"Failed to clean up workspace {self.workspace}: {e}")
        self.workspace = None

    def is_expired(self) -> bool:
        """Check if workspace has exceeded max age."""
        if self.created_at is None:
            return False
        return datetime.now() - self.created_at > self.max_age

    @staticmethod
    def cleanup_old_workspaces(
        base_dir: Path,
        prefix: str = "retail_analytics_",
        max_age_hours: int = 24,
    ) -> int:
        """Clean up old workspace directories."""
        cleaned = 0
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith(prefix):
                try:
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        shutil.rmtree(item, ignore_errors=True)
                        cleaned += 1
                except Exception:
                    pass

        return cleaned


@contextmanager
def secure_temp_dir(
    prefix: str = "retail_analytics_",
    max_age_hours: int = 24,
) -> Path:
    """Context manager for secure temporary directory."""
    workspace = SecureWorkspace(prefix=prefix, max_age_hours=max_age_hours)
    path = workspace.__enter__()
    try:
        yield path
    finally:
        workspace.__exit__(None, None, None)


# ============================================================================
# CSV Export Security
# ============================================================================

FORMULA_INJECTION_PREFIXES = {'=', '+', '-', '@', '\t', '\r'}


def escape_csv_value(value: str) -> str:
    """
    Escape value for CSV export to prevent formula injection.
    
    Prepends a single quote to values starting with formula characters.
    """
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if value and value[0] in FORMULA_INJECTION_PREFIXES:
        return "'" + value
    return value


def escape_csv_row(row: dict[str, Any]) -> dict[str, str]:
    """Escape all values in a row for CSV export."""
    return {k: escape_csv_value(str(v)) if v is not None else "" for k, v in row.items()}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe download.
    
    Removes path components and dangerous characters.
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Remove dangerous characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    filename = ''.join(c for c in filename if c in safe_chars)

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename or "download"


# ============================================================================
# Data Privacy Utilities
# ============================================================================

def mask_customer_id(customer_id: str, salt: str = "") -> str:
    """
    Create a masked/hased customer ID for display.
    
    Uses deterministic hashing with optional salt.
    """
    if not customer_id:
        return "UNKNOWN"

    hasher = hashlib.sha256()
    hasher.update(f"{salt}{customer_id}".encode())
    return f"CUST_{hasher.hexdigest()[:8].upper()}"


def mask_pii_fields(data: dict[str, Any], pii_fields: list[str]) -> dict[str, Any]:
    """
    Mask PII fields in a data dictionary.
    
    Replaces values with masked versions.
    """
    result = data.copy()
    for field in pii_fields:
        if field in result:
            if field in ('customer_id', 'customer_email', 'customer_name'):
                result[field] = mask_customer_id(str(result[field]))
            elif field in ('phone', 'address', 'zip_code'):
                result[field] = "[REDACTED]"
            else:
                result[field] = "[MASKED]"
    return result


def check_pii_exposure(data: list[dict], pii_fields: list[str]) -> list[str]:
    """
    Check data for potential PII exposure.
    
    Returns list of warnings.
    """
    warnings = []
    for i, record in enumerate(data):
        for field in pii_fields:
            if field in record and record[field] is not None:
                value = str(record[field])
                # Check for common PII patterns
                if field == 'email' and '@' in value:
                    warnings.append(f"Record {i}: Email address in '{field}' field")
                elif field == 'phone' and any(c.isdigit() for c in value):
                    warnings.append(f"Record {i}: Phone number in '{field}' field")
    return warnings


# ============================================================================
# Access Control
# ============================================================================

class AccessControl:
    """Simple role-based access control for dashboard features."""

    ROLES = {
        'viewer': {'view_dashboard', 'view_reports', 'export_csv'},
        'analyst': {'view_dashboard', 'view_reports', 'export_csv', 'run_analysis', 'view_customers'},
        'admin': {'view_dashboard', 'view_reports', 'export_csv', 'run_analysis',
                  'view_customers', 'manage_users', 'configure_system'},
    }

    def __init__(self, role: str = 'viewer'):
        self.role = role
        self.permissions = self.ROLES.get(role, set())

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions

    def require_permission(self, permission: str) -> None:
        if not self.has_permission(permission):
            raise PermissionError(f"Role '{self.role}' lacks permission: {permission}")


# ============================================================================
# Audit Logging
# ============================================================================

AUDIT_LOG_PATH = Path.home() / ".retail_analytics" / "audit.log"


def audit_log(
    event: str,
    user: str | None = None,
    resource: str | None = None,
    action: str | None = None,
    outcome: str = "success",
    details: dict | None = None,
) -> None:
    """
    Log security-relevant events for audit trail.
    
    Events: login, logout, upload, download, analysis_run, config_change, access_denied
    """
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "user": user or os.getenv("USER", "unknown"),
        "resource": resource,
        "action": action,
        "outcome": outcome,
        "details": details or {},
    }

    try:
        with AUDIT_LOG_PATH.open('a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        # Don't fail the main operation if audit logging fails
        pass


# ============================================================================
# Content Security Policy Helpers
# ============================================================================

CSP_HEADER = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.plot.ly; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


def get_csp_header() -> str:
    """Get Content Security Policy header value."""
    return CSP_HEADER


# ============================================================================
# Security Configuration Validation
# ============================================================================

def validate_deployment_config(config: dict[str, Any]) -> list[str]:
    """
    Validate deployment configuration for security issues.
    
    Returns list of warnings/errors.
    """
    issues = []

    # Check for debug mode in production
    if config.get('debug', False):
        issues.append("ERROR: Debug mode enabled in production")

    # Check for default secrets
    secret_keys = ['secret_key', 'api_key', 'password', 'token']
    for key in secret_keys:
        value = config.get(key, '')
        if value in ('changeme', 'secret', 'password', 'admin', '', 'default'):
            issues.append(f"ERROR: Default/insecure value for {key}")

    # Check HTTPS enforcement
    if not config.get('force_https', True):
        issues.append("WARNING: HTTPS not enforced")

    # Check session security
    if not config.get('session_secure', True):
        issues.append("WARNING: Secure session cookies not enforced")

    if not config.get('session_httponly', True):
        issues.append("WARNING: HttpOnly session cookies not enforced")

    # Check CORS
    cors_origin = config.get('cors_origin', '*')
    if cors_origin == '*':
        issues.append("WARNING: CORS allows all origins")

    return issues


# ============================================================================
# Export all
# ============================================================================

__all__ = [
    # Upload validation
    "validate_upload_file", "validate_parquet_file", "validate_csv_file",
    "MAX_UPLOAD_SIZE", "ALLOWED_EXTENSIONS",
    # Secure workspace
    "SecureWorkspace", "secure_temp_dir",
    # CSV export security
    "escape_csv_value", "escape_csv_row", "sanitize_filename",
    "FORMULA_INJECTION_PREFIXES",
    # Data privacy
    "mask_customer_id", "mask_pii_fields", "check_pii_exposure",
    # Access control
    "AccessControl",
    # Audit logging
    "audit_log", "AUDIT_LOG_PATH",
    # CSP
    "get_csp_header", "CSP_HEADER",
    # Config validation
    "validate_deployment_config",
]
