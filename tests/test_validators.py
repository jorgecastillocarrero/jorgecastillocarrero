"""
Tests for src/validators.py - Input validation utilities.
"""

import pytest

from src.validators import (
    ValidationError,
    validate_symbol,
    validate_positive_number,
    validate_positive_int,
    validate_currency,
    validate_account_code,
)


class TestValidateSymbol:
    """Tests for validate_symbol function."""

    def test_simple_us_symbol(self):
        """Test simple US symbol without exchange."""
        code, exchange = validate_symbol("AAPL")
        assert code == "AAPL"
        assert exchange == "US"

    def test_symbol_with_exchange(self):
        """Test symbol with exchange suffix."""
        code, exchange = validate_symbol("IAG.MC")
        assert code == "IAG"
        assert exchange == "MC"

    def test_lowercase_converted(self):
        """Test that lowercase is converted to uppercase."""
        code, exchange = validate_symbol("aapl.us")
        assert code == "AAPL"
        assert exchange == "US"

    def test_forex_pair(self):
        """Test forex pair format."""
        code, exchange = validate_symbol("EURUSD=X")
        assert code == "EURUSD=X"
        assert exchange == "US"

    def test_symbol_with_dash(self):
        """Test symbol with dash (e.g., BRK-B)."""
        code, exchange = validate_symbol("BRK-B")
        assert code == "BRK-B"
        assert exchange == "US"

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed."""
        code, exchange = validate_symbol("  AAPL  ")
        assert code == "AAPL"

    def test_empty_raises(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol("")

    def test_none_raises(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol(None)

    def test_invalid_characters_raises(self):
        """Test that invalid characters raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol("AAPL;DROP")

    def test_too_many_dots_raises(self):
        """Test that too many dots raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol("AAPL.US.EXTRA")

    def test_empty_code_raises(self):
        """Test that empty code part raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol(".US")

    def test_empty_exchange_raises(self):
        """Test that empty exchange part raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol("AAPL.")

    def test_too_long_raises(self):
        """Test that too long symbol raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_symbol("A" * 25)


class TestValidatePositiveNumber:
    """Tests for validate_positive_number function."""

    def test_valid_positive(self):
        """Test valid positive number."""
        assert validate_positive_number(100.5, "amount") == 100.5

    def test_valid_integer(self):
        """Test valid positive integer."""
        assert validate_positive_number(100, "amount") == 100

    def test_zero_not_allowed_by_default(self):
        """Test that zero raises by default."""
        with pytest.raises(ValidationError):
            validate_positive_number(0, "amount")

    def test_zero_allowed_when_specified(self):
        """Test that zero is allowed when specified."""
        assert validate_positive_number(0, "amount", allow_zero=True) == 0

    def test_negative_raises(self):
        """Test that negative number raises."""
        with pytest.raises(ValidationError):
            validate_positive_number(-10, "amount")

    def test_min_value(self):
        """Test minimum value check."""
        assert validate_positive_number(10, "amount", min_value=5) == 10
        with pytest.raises(ValidationError):
            validate_positive_number(3, "amount", min_value=5)

    def test_max_value(self):
        """Test maximum value check."""
        assert validate_positive_number(50, "amount", max_value=100) == 50
        with pytest.raises(ValidationError):
            validate_positive_number(150, "amount", max_value=100)

    def test_string_raises(self):
        """Test that string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_positive_number("100", "amount")


class TestValidatePositiveInt:
    """Tests for validate_positive_int function."""

    def test_valid_positive(self):
        """Test valid positive integer."""
        assert validate_positive_int(100, "days") == 100

    def test_zero_raises(self):
        """Test that zero raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_positive_int(0, "days")

    def test_negative_raises(self):
        """Test that negative raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_positive_int(-5, "days")

    def test_float_raises(self):
        """Test that float raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_positive_int(10.5, "days")

    def test_max_value(self):
        """Test maximum value check."""
        assert validate_positive_int(100, "days", max_value=365) == 100
        with pytest.raises(ValidationError):
            validate_positive_int(500, "days", max_value=365)

    def test_min_value(self):
        """Test minimum value check."""
        assert validate_positive_int(10, "days", min_value=5) == 10
        with pytest.raises(ValidationError):
            validate_positive_int(3, "days", min_value=5)

    def test_bool_raises(self):
        """Test that boolean raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_positive_int(True, "days")


class TestValidateCurrency:
    """Tests for validate_currency function."""

    def test_valid_eur(self):
        """Test valid EUR currency."""
        assert validate_currency("EUR") == "EUR"

    def test_valid_usd(self):
        """Test valid USD currency."""
        assert validate_currency("USD") == "USD"

    def test_lowercase_converted(self):
        """Test that lowercase is converted."""
        assert validate_currency("eur") == "EUR"

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed."""
        assert validate_currency("  USD  ") == "USD"

    def test_invalid_currency_raises(self):
        """Test that invalid currency raises."""
        with pytest.raises(ValidationError):
            validate_currency("XXX")

    def test_empty_raises(self):
        """Test that empty string raises."""
        with pytest.raises(ValidationError):
            validate_currency("")

    def test_none_raises(self):
        """Test that None raises."""
        with pytest.raises(ValidationError):
            validate_currency(None)


class TestValidateAccountCode:
    """Tests for validate_account_code function."""

    def test_valid_ib(self):
        """Test valid IB account code."""
        assert validate_account_code("IB") == "IB"

    def test_valid_alphanumeric(self):
        """Test valid alphanumeric account code."""
        assert validate_account_code("CO3365") == "CO3365"

    def test_lowercase_converted(self):
        """Test that lowercase is converted."""
        assert validate_account_code("ib") == "IB"

    def test_with_underscore(self):
        """Test account code with underscore."""
        assert validate_account_code("IB_MAIN") == "IB_MAIN"

    def test_with_dash(self):
        """Test account code with dash."""
        assert validate_account_code("IB-2") == "IB-2"

    def test_empty_raises(self):
        """Test that empty string raises."""
        with pytest.raises(ValidationError):
            validate_account_code("")

    def test_invalid_characters_raises(self):
        """Test that invalid characters raise."""
        with pytest.raises(ValidationError):
            validate_account_code("IB@MAIN")

    def test_too_long_raises(self):
        """Test that too long code raises."""
        with pytest.raises(ValidationError):
            validate_account_code("A" * 25)
