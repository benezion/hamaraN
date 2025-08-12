#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

def is_number_only_1_to_99(text):
    if not text:
        return False
    text_str = str(text).strip()
    if text_str.isdigit():
        num = int(text_str)
        return 1 <= num <= 99
    return False

def is_year_4_digits(text):
    if not text:
        return False
    text_str = str(text).strip()
    if text_str.isdigit() and len(text_str) == 4:
        year = int(text_str)
        return 1900 <= year <= 2100
    return False

def is_date_pattern(text):
    if not text:
        return False
    text_str = str(text).strip()
    date_patterns = [
        r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$',
        r'^\d{2,4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}$',
        r'^\d{1,2}[/\-\.]\d{2,4}$',
        r'^\d{2,4}[/\-\.]\d{1,2}$',
    ]
    for pattern in date_patterns:
        if re.match(pattern, text_str):
            return True
    return False

def is_month_name(text):
    if not text:
        return False
    text_str = str(text).strip().lower()
    hebrew_months = [
        'ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני',
        'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר',
        'תשרי', 'חשון', 'כסלו', 'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיון',
        'תמוז', 'אב', 'אלול', 'אדר א', 'אדר ב'
    ]
    return text_str in hebrew_months

def extract_month_from_text(text):
    if not text:
        return None
    text_str = str(text).strip().lower()
    hebrew_months = [
        'ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני',
        'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר',
        'תשרי', 'חשון', 'כסלו', 'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיון',
        'תמוז', 'אב', 'אלול', 'אדר א', 'אדר ב'
    ]
    # Check if it's a clean month name
    if text_str in hebrew_months:
        return text_str
    # Check if it contains a month with prefix (like באפריל, במרץ, ו-מאי)
    for month in hebrew_months:
        if month in text_str:
            return month
    return None

def is_day_with_prefix(text):
    if not text:
        return False
    text_str = str(text).strip()
    match = re.match(r'^([מבו])\-(\d{1,2})$', text_str)
    if match:
        day = int(match.group(2))
        return 1 <= day <= 31
    return False

def is_day_number_1_to_31(text):
    """Check if text is a day number (1-31) without prefix"""
    if not text:
        return False
    text_str = str(text).strip()
    if text_str.isdigit():
        day = int(text_str)
        return 1 <= day <= 31
    return False

def is_month_with_b_prefix(text):
    """Check if text is a month name with ב prefix (like בינואר, בפברואר)"""
    if not text:
        return False
    text_str = str(text).strip().lower()

    # Pattern: ב + month name
    if text_str.startswith('ב'):
        month_part = text_str[1:]  # Remove the ב prefix
        hebrew_months = [
            'ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני',
            'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר',
            'תשרי', 'חשון', 'כסלו', 'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיון',
            'תמוז', 'אב', 'אלול', 'אדר א', 'אדר ב'
        ]
        return month_part in hebrew_months
    return False

def is_month_with_prefix_or_contains_month(text):
    """Check if text contains a month name (with or without prefix)"""
    if not text:
        return False

    # First check if it's a clean month name
    if is_month_name(text):
        return True

    # Then check if it contains a month name anywhere in the text
    return extract_month_from_text(text) is not None

def is_number(text):
    if not text:
        return False
    text_str = str(text).strip()
    if text_str.isdigit():
        return True
    try:
        normalized = text_str.replace(',', '.')
        float(normalized)
        return True
    except ValueError:
        return False

def extract_prefix_pattern(text):
    if not text:
        return None
    text_str = str(text).strip()
    match = re.match(r'^([^\d]+\-)', text_str)
    if match:
        return match.group(1)
    return None

def should_exclude_column_c(text):
    return (is_number_only_1_to_99(text) or
            is_year_4_digits(text) or
            is_date_pattern(text) or
            is_month_name(text))

def contains_non_hebrew_letters(text):
    if not text:
        return False
    text_str = str(text)
    for char in text_str:
        if char in ' .,!?;:()[]{}"-\'':
            continue
        if not ('\u05D0' <= char <= '\u05EA'):
            return True
    return False

def filter_unique_currency_patterns(df, second_column, third_column, currency_term):
    currency_rows = df[df[third_column] == currency_term].copy()
    if len(currency_rows) == 0:
        return df, 0, 0, 0

    currency_rows['prefix_pattern'] = currency_rows[second_column].apply(extract_prefix_pattern)
    with_prefix = currency_rows[currency_rows['prefix_pattern'].notna()]
    without_prefix = currency_rows[currency_rows['prefix_pattern'].isna()]
    unique_prefix_rows = with_prefix.drop_duplicates(subset=['prefix_pattern'], keep='first')

    print(f"  {currency_term} rows with prefix patterns: {len(with_prefix):,}")
    print(f"  Unique prefix patterns kept: {len(unique_prefix_rows):,}")

    if len(unique_prefix_rows) > 0:
        print(f"  Examples of unique prefix patterns kept for {currency_term}:")
        for idx, (_, row) in enumerate(unique_prefix_rows.head(3).iterrows()):
            print(f"    {row['prefix_pattern']}: B='{row[second_column]}', C='{row[third_column]}'")

    excluded_pure_numbers = without_prefix[without_prefix[second_column].apply(is_number)]
    kept_without_prefix = without_prefix[~without_prefix[second_column].apply(is_number)]

    if len(excluded_pure_numbers) > 0:
        print(f"  {currency_term} rows with pure numbers excluded: {len(excluded_pure_numbers):,}")

    kept_currency_rows = pd.concat([unique_prefix_rows, kept_without_prefix], ignore_index=True)
    other_rows = df[df[third_column] != currency_term]
    result_df = pd.concat([other_rows, kept_currency_rows], ignore_index=True)

    return result_df, len(with_prefix), len(unique_prefix_rows), len(excluded_pure_numbers)

def filter_mechulek_unique(df, second_column):
    mechulek_rows = df[df[second_column].str.contains('מחולק', na=False)].copy()
    if len(mechulek_rows) == 0:
        return df, 0, 0

    kept_mechulek = mechulek_rows.head(1)
    print(f"  מחולק rows found: {len(mechulek_rows):,}")
    print(f"  מחולק rows kept: {len(kept_mechulek):,}")
    print(f"  מחולק rows excluded: {len(mechulek_rows) - len(kept_mechulek):,}")

    if len(kept_mechulek) > 0:
        row = kept_mechulek.iloc[0]
        print(f"  Example kept: A='{row[df.columns[0]]}', B='{row[second_column]}', C='{row[df.columns[2]]}'")

    other_rows = df[~df[second_column].str.contains('מחולק', na=False)]
    result_df = pd.concat([other_rows, kept_mechulek], ignore_index=True)
    return result_df, len(mechulek_rows), len(mechulek_rows) - len(kept_mechulek)

def filter_date_combinations(df, second_column, third_column):
    """For date combinations with prefix, keep only 1 line with a number for each month + all different prefix formats for that month in Column C"""
    date_rows = df[
        df[second_column].apply(is_day_with_prefix) &
        df[third_column].apply(is_month_with_prefix_or_contains_month)
    ].copy()

    if len(date_rows) == 0:
        return df, 0, 0

    print(f"  Date combination rows (with prefix) found: {len(date_rows):,}")
    date_rows['clean_month'] = date_rows[third_column].apply(extract_month_from_text)

    kept_rows = []
    months_processed = set()

    for month in date_rows['clean_month'].unique():
        if month is None:
            continue
        month_rows = date_rows[date_rows['clean_month'] == month]
        first_number_row = month_rows.head(1)
        kept_rows.extend(first_number_row.to_dict('records'))
        months_processed.add(month)

    if kept_rows:
        kept_date_rows = pd.DataFrame(kept_rows)
        kept_date_rows = kept_date_rows.drop('clean_month', axis=1)
    else:
        kept_date_rows = pd.DataFrame()

    print(f"  Months processed (with prefix): {len(months_processed):,}")
    print(f"  Date rows kept (with prefix): {len(kept_date_rows):,}")
    print(f"  Date rows excluded (with prefix): {len(date_rows) - len(kept_date_rows):,}")

    if len(kept_date_rows) > 0:
        print(f"  Examples of date combinations kept (with prefix):")
        for idx, (_, row) in enumerate(kept_date_rows.head(3).iterrows()):
            print(f"    B='{row[second_column]}', C='{row[third_column]}'")

    other_rows = df[~(
        df[second_column].apply(is_day_with_prefix) &
        df[third_column].apply(is_month_with_prefix_or_contains_month)
    )]

    if len(kept_date_rows) > 0:
        result_df = pd.concat([other_rows, kept_date_rows], ignore_index=True)
    else:
        result_df = other_rows

    return result_df, len(date_rows), len(date_rows) - len(kept_date_rows)

def filter_number_month_b_combinations(df, second_column, third_column):
    """
    For combinations where:
    - Column B has number 1-31 (without prefix)
    - Column C has month name with ב prefix (like בינואר, בפברואר)

    Keep only 1 line for each month
    """
    date_rows = df[
        df[second_column].apply(is_day_number_1_to_31) &
        df[third_column].apply(is_month_with_b_prefix)
    ].copy()

    if len(date_rows) == 0:
        return df, 0, 0

    print(f"  Number + ב-month combination rows found: {len(date_rows):,}")

    # Extract the clean month name for grouping (remove ב prefix)
    date_rows['clean_month'] = date_rows[third_column].apply(lambda x: str(x).strip().lower()[1:] if str(x).strip().lower().startswith('ב') else None)

    kept_rows = []
    months_processed = set()

    for month in date_rows['clean_month'].unique():
        if month is None:
            continue
        month_rows = date_rows[date_rows['clean_month'] == month]
        first_number_row = month_rows.head(1)
        kept_rows.extend(first_number_row.to_dict('records'))
        months_processed.add(month)

    if kept_rows:
        kept_date_rows = pd.DataFrame(kept_rows)
        kept_date_rows = kept_date_rows.drop('clean_month', axis=1)
    else:
        kept_date_rows = pd.DataFrame()

    print(f"  ב-months processed: {len(months_processed):,}")
    print(f"  Number + ב-month rows kept: {len(kept_date_rows):,}")
    print(f"  Number + ב-month rows excluded: {len(date_rows) - len(kept_date_rows):,}")

    if len(kept_date_rows) > 0:
        print(f"  Examples of number + ב-month combinations kept:")
        for idx, (_, row) in enumerate(kept_date_rows.head(5).iterrows()):
            print(f"    B='{row[second_column]}', C='{row[third_column]}'")

    # Remove original date rows and add back the filtered ones
    other_rows = df[~(
        df[second_column].apply(is_day_number_1_to_31) &
        df[third_column].apply(is_month_with_b_prefix)
    )]

    if len(kept_date_rows) > 0:
        result_df = pd.concat([other_rows, kept_date_rows], ignore_index=True)
    else:
        result_df = other_rows

    return result_df, len(date_rows), len(date_rows) - len(kept_date_rows)

def create_formatted_excel(df, output_file):
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Word Combinations"
        ws.sheet_view.rightToLeft = True

        header_font = Font(name='Arial', size=12, bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')
        header_alignment = Alignment(horizontal='center', vertical='center')
        data_font = Font(name='Arial', size=11)
        data_alignment = Alignment(horizontal='right', vertical='center', wrap_text=True)
        border = Border(
            left=Side(border_style='thin', color='000000'),
            right=Side(border_style='thin', color='000000'),
            top=Side(border_style='thin', color='000000'),
            bottom=Side(border_style='thin', color='000000')
        )

        headers = ['Word_Pair', 'First_Word', 'Second_Word', 'Context']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = border

        for row_idx, (_, row) in enumerate(df.iterrows(), 2):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.font = data_font
                if col_idx == 1:
                    special_alignment = Alignment(horizontal='right', vertical='center', wrap_text=True)
                    cell.alignment = special_alignment
                else:
                    cell.alignment = data_alignment
                cell.border = border

        for col in range(1, 5):
            column_letter = chr(64 + col)
            max_length = 0
            for row in range(1, min(ws.max_row + 1, 100)):
                cell_value = ws.cell(row=row, column=col).value
                if cell_value:
                    cell_length = len(str(cell_value))
                    if cell_length > max_length:
                        max_length = cell_length
            adjusted_width = min(max_length + 5, 60)
            ws.column_dimensions[column_letter].width = adjusted_width

        ws.auto_filter.ref = f"A1:D{ws.max_row}"
        wb.save(output_file)
        return True
    except Exception as e:
        print(f"Error creating formatted Excel: {str(e)}")
        return False

def filter_excel_final_with_month_b(input_file, output_file):
    print(f"Reading Excel file: {input_file}")

    try:
        df = pd.read_excel(input_file)
        print(f"Original file loaded successfully.")
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        first_column = df.columns[0]
        second_column = df.columns[1]
        third_column = df.columns[2]

        print(f"\nApplying comprehensive filters including number + ב-month combinations:")
        print(f"1. Column A ({first_column}): Keep rows with non-Hebrew characters")
        print(f"2. Column C ({third_column}): Exclude numbers 1-99")
        print(f"3. Column C ({third_column}): Exclude 4-digit years")
        print(f"4. Column C ({third_column}): Exclude date patterns")
        print(f"5. Column C ({third_column}): Exclude month names")
        print(f"6. Refined currency filter for BOTH שקלים and ש\"ח:")
        print(f"   - Keep only one unique format of each prefix pattern")
        print(f"   - Exclude pure number combinations")
        print(f"7. Column B ({second_column}): Keep only one מחולק occurrence")
        print(f"8. Date combinations (with prefix): Keep 1 number per month + all prefix formats")
        print(f"9. Number + ב-month combinations: Keep 1 line per month")

        step1_df = df[df[first_column].apply(contains_non_hebrew_letters)]
        print(f"\nAfter step 1 (non-Hebrew filter): {len(step1_df):,} rows")

        step2_df = step1_df[~step1_df[third_column].apply(should_exclude_column_c)]
        print(f"After step 2 (Column C filters): {len(step2_df):,} rows")

        print(f"\nApplying unique prefix pattern filtering for שקלים:")
        step3_df, shekel_with_prefix, shekel_unique, shekel_excluded = filter_unique_currency_patterns(
            step2_df, second_column, third_column, 'שקלים'
        )

        print(f"\nApplying unique prefix pattern filtering for ש\"ח:")
        step4_df, shach_with_prefix, shach_unique, shach_excluded = filter_unique_currency_patterns(
            step3_df, second_column, third_column, 'ש"ח'
        )

        print(f"\nApplying מחולק unique filtering:")
        step5_df, mechulek_found, mechulek_excluded = filter_mechulek_unique(step4_df, second_column)

        print(f"\nApplying date combinations filtering (with prefix):")
        step6_df, date_found, date_excluded = filter_date_combinations(step5_df, second_column, third_column)

        print(f"\nApplying number + ב-month combinations filtering:")
        filtered_df, month_b_found, month_b_excluded = filter_number_month_b_combinations(step6_df, second_column, third_column)

        print(f"\nFinal filtering results:")
        print(f"Original rows: {len(df):,}")
        print(f"After non-Hebrew filter: {len(step1_df):,}")
        print(f"After Column C filters: {len(step2_df):,}")
        print(f"After שקלים unique patterns: {len(step3_df):,}")
        print(f"After ש\"ח unique patterns: {len(step4_df):,}")
        print(f"After מחולק unique filter: {len(step5_df):,}")
        print(f"After date combinations filter: {len(step6_df):,}")
        print(f"After number + ב-month filter: {len(filtered_df):,}")
        print(f"Total removed: {len(df) - len(filtered_df):,}")

        if len(filtered_df) == 0:
            print("No rows found after applying all filters!")
            return None

        print(f"\nSample of filtered data (first 5 rows):")
        for idx, (_, row) in enumerate(filtered_df.head(5).iterrows()):
            print(f"Row {idx}:")
            for col in df.columns:
                print(f"  {col}: {row[col]}")
            print("-" * 40)

        success = create_formatted_excel(filtered_df, output_file)

        if success:
            print(f"\nFormatted Excel file created successfully: {output_file}")
        else:
            filtered_df.to_excel(output_file, index=False)
            print(f"Basic Excel file created: {output_file}")

        print("\n" + "="*70)
        print("COMPREHENSIVE FILTERING STATISTICS (INCLUDING NUMBER + ב-MONTH)")
        print("="*70)
        print(f"Original rows: {len(df):,}")
        print(f"After non-Hebrew filter: {len(step1_df):,}")
        print(f"After Column C filters: {len(step2_df):,}")
        print(f"שקלים - with prefix patterns: {shekel_with_prefix:,}")
        print(f"שקלים - unique patterns kept: {shekel_unique:,}")
        print(f"שקלים - pure numbers excluded: {shekel_excluded:,}")
        print(f"ש\"ח - with prefix patterns: {shach_with_prefix:,}")
        print(f"ש\"ח - unique patterns kept: {shach_unique:,}")
        print(f"ש\"ח - pure numbers excluded: {shach_excluded:,}")
        print(f"מחולק - total found: {mechulek_found:,}")
        print(f"מחולק - excluded: {mechulek_excluded:,}")
        print(f"Date combinations (with prefix) - found: {date_found:,}")
        print(f"Date combinations (with prefix) - excluded: {date_excluded:,}")
        print(f"Number + ב-month combinations - found: {month_b_found:,}")
        print(f"Number + ב-month combinations - excluded: {month_b_excluded:,}")
        print(f"Final filtered rows: {len(filtered_df):,}")
        print(f"Total removed: {len(df) - len(filtered_df):,}")
        print(f"Output file created: {output_file}")
        print("="*70)

        return {
            'original_rows': len(df),
            'final_rows': len(filtered_df),
            'total_removed': len(df) - len(filtered_df),
            'month_b_stats': {'found': month_b_found, 'excluded': month_b_excluded}
        }

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    input_file = "all_2_words_fixed.xlsx"
    output_file = "all_2_words_filtered_final_complete_all.xlsx"

    print("FINAL COMPREHENSIVE Excel Filter (Including Number + ב-Month)")
    print("Applies ALL filters:")
    print("1. Keep rows where Column A contains non-Hebrew characters (א-ת)")
    print("2. Exclude rows where Column C is only a number 1-99")
    print("3. Exclude rows where Column C is a 4-digit year")
    print("4. Exclude rows where Column C is a date pattern")
    print("5. Exclude rows where Column C is a month name (Hebrew/English)")
    print("6. Refined currency filter for BOTH שקלים and ש\"ח:")
    print("   - Keep only one unique format of each prefix pattern (ו-, ומ-, etc.)")
    print("   - Exclude pure number combinations")
    print("7. Keep only ONE occurrence of מחולק in Column B")
    print("8. Date combinations filter (with prefix):")
    print("   - Column B: number 1-31 + prefix (מ/ב/ו) + \"-\"")
    print("   - Column C: contains month name (like באפריל, במרץ)")
    print("   - Keep only 1 number per month + all prefix formats")
    print("9. Number + ב-month combinations filter:")
    print("   - Column B: number 1-31 (without prefix)")
    print("   - Column C: month name with ב prefix (like בינואר, בפברואר)")
    print("   - Keep only 1 line per month")
    print("Features: RTL direction, professional formatting, column filters")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("-" * 70)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist in the current directory.")
        return

    stats = filter_excel_final_with_month_b(input_file, output_file)

    if stats:
        print("\nFormatting features applied:")
        print("✓ Right-to-left (RTL) direction")
        print("✓ Professional header styling")
        print("✓ Auto-adjusted column widths")
        print("✓ Filter buttons on columns A, B, C, D")
        print("✓ Cell borders and alignment")
        print("✓ Non-Hebrew character filtering (Column A)")
        print("✓ Numbers 1-99 exclusion (Column C)")
        print("✓ 4-digit years exclusion (Column C)")
        print("✓ Date patterns exclusion (Column C)")
        print("✓ Month names exclusion (Column C)")
        print("✓ Refined currency filtering (unique prefix patterns for BOTH שקלים and ש\"ח)")
        print("✓ מחולק unique occurrence (Column B)")
        print("✓ Date combinations filtering (1 number per month + all prefix formats)")
        print("✓ Number + ב-month combinations filtering (1 line per month)")
        print("✓ All columns preserved (A, B, C, D)")

if __name__ == "__main__":
    main()
