# -*- coding: utf-8 -*-
"""ConsolidatedDateProcessor extracted from translationLobe.py"""
from consts import HEBREW_MONTHS

class ConsolidatedDateProcessor:
    """Unified date handling utilities"""
    def __init__(self, text_processor, hebrew_months, heb2num_func):
        self.text_processor = text_processor
        self._hebrew_months = hebrew_months
        self.heb2num = heb2num_func

    def convert_day_to_hebrew(self, day_num: int) -> str:
        if 1 <= day_num <= 10:
            return self.text_processor.ORDINAL_MASCULINE[day_num]
        return self.heb2num(day_num, 'm')

    def convert_month_to_hebrew(self, month_num: int) -> str:
        return self._hebrew_months.get(month_num, str(month_num))

    def convert_year_to_hebrew(self, year_num: int) -> str:
        return self.heb2num(year_num, 'f', is_year=True)

    def validate_date_components(self, day: int, month: int, year: int | None = None) -> bool:
        return (1 <= day <= 31) and (1 <= month <= 12) and (year is None or 1900 <= year <= 2100)

    def format_hebrew_date(self, day_text: str, month_name: str, year_text: str | None = None, prefix: str = "") -> str:
        return f"{prefix}{day_text} ב{month_name}{' ' + year_text if year_text else ''}"

    def convert_date_pattern(self, day_str: str, month_str: str, year_str: str | None = None, prefix: str = "") -> str | None:
        try:
            day, month = int(day_str), int(month_str)
            year = int(year_str) if year_str else None
            if not self.validate_date_components(day, month, year):
                return None
            day_txt = self.convert_day_to_hebrew(day)
            month_txt = self.convert_month_to_hebrew(month)
            year_txt = self.convert_year_to_hebrew(year) if year else None
            return self.format_hebrew_date(day_txt, month_txt, year_txt, prefix)
        except ValueError:
            return None
