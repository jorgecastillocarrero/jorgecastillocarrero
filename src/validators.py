"""
Input validation utilities.
Provides validation functions for common inputs across the application.
"""

import re
from typing import Optional, Tuple


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_symbol(symbol: str) -> Tuple[str, str]:
    """
    Validate and parse a stock symbol string.

    Args:
        symbol: Symbol in format "CODE" or "CODE.EXCHANGE"

    Returns:
        Tuple of (symbol_code, exchange_code)

    Raises:
        ValidationError: If symbol format is invalid

    Examples:
        >>> validate_symbol("AAPL")
        ('AAPL', 'US')
        >>> validate_symbol("IAG.MC")
        ('IAG', 'MC')
        >>> validate_symbol("EURUSD=X")
        ('EURUSD=X', 'US')
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError(f"Invalid symbol: {symbol!r} - must be a non-empty string")

    symbol = symbol.strip().upper()

    if len(symbol) > 20:
        raise ValidationError(f"Symbol too long: {symbol} (max 20 characters)")

    # Allow alphanumeric, dash, dot, and equals (for forex pairs like EURUSD=X)
    if not re.match(r'^[A-Z0-9\-\.=]+$', symbol):
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")

    # Handle forex pairs (EURUSD=X)
    if '=' in symbol:
        return symbol, 'US'

    parts = symbol.split(".")
    if len(parts) == 1:
        return parts[0], "US"
    elif len(parts) == 2:
        if not parts[0] or not parts[1]:
            raise ValidationError(f"Invalid symbol format: {symbol}")
        return parts[0], parts[1]
    else:
        raise ValidationError(f"Invalid symbol format (too many dots): {symbol}")


def validate_positive_number(
    value: float,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = False
) -> float:
    """
    Validate a positive number parameter.

    Args:
        value: The value to validate
        name: Parameter name (for error messages)
        min_value: Optional minimum value
        max_value: Optional maximum value
        allow_zero: Whether zero is allowed (default False)

    Returns:
        The validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got: {type(value).__name__}")

    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got: {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got: {value}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be at least {min_value}, got: {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be at most {max_value}, got: {value}")

    return value


def validate_positive_int(
    value: int,
    name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> int:
    """
    Validate a positive integer parameter.

    Args:
        value: The value to validate
        name: Parameter name (for error messages)
        min_value: Optional minimum value
        max_value: Optional maximum value

    Returns:
        The validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got: {type(value).__name__}")

    if value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got: {value}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be at least {min_value}, got: {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} exceeds maximum value {max_value}: {value}")

    return value


def validate_currency(currency: str) -> str:
    """
    Validate a currency code.

    Args:
        currency: Currency code (e.g., 'EUR', 'USD')

    Returns:
        Uppercase currency code

    Raises:
        ValidationError: If currency code is invalid
    """
    if not currency or not isinstance(currency, str):
        raise ValidationError(f"Invalid currency: {currency!r}")

    currency = currency.strip().upper()

    # Standard 3-letter currency codes
    valid_currencies = {'EUR', 'USD', 'GBP', 'CAD', 'CHF', 'JPY', 'AUD', 'NZD', 'SEK', 'NOK', 'DKK'}

    if currency not in valid_currencies:
        raise ValidationError(f"Unsupported currency: {currency}. Valid: {', '.join(sorted(valid_currencies))}")

    return currency


def validate_account_code(account_code: str) -> str:
    """
    Validate an account code.

    Args:
        account_code: Account code (e.g., 'IB', 'CO3365')

    Returns:
        Uppercase account code

    Raises:
        ValidationError: If account code is invalid
    """
    if not account_code or not isinstance(account_code, str):
        raise ValidationError(f"Invalid account code: {account_code!r}")

    account_code = account_code.strip().upper()

    if len(account_code) > 20:
        raise ValidationError(f"Account code too long: {account_code}")

    if not re.match(r'^[A-Z0-9_\-]+$', account_code):
        raise ValidationError(f"Account code contains invalid characters: {account_code}")

    return account_code
