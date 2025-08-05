# -*- coding: utf-8 -*-
"""Table utilities extracted from translationLobe.py"""
import os
import pandas as pd

class TextProcessingTable:
    """Table-based text processor for Hebrew text processing (extracted)"""

    def __init__(self, text, debug: bool = False):
        self.original_text = text
        self.debug = debug
        self.rows = []
        self.line_boundaries = []  # Track where each line ends

        # Initialize table with one row per word, preserving line structure
        lines = text.split('\n')
        word_index = 1
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            words = line.split()
            line_start_word = word_index
            for word in words:
                self.rows.append({
                    'row': word_index,
                    'source': word,
                    'gender': None,
                    'gender_source': None,
                    'person': None,
                    'milon': None,
                    'heb2num': None,
                    'pattern': None,
                    'consumed': None,
                    'processor': None,
                    'notes': '',
                    'line_num': line_num + 1
                })
                word_index += 1

            if words:
                self.line_boundaries.append({
                    'line_num': line_num + 1,
                    'start_row': line_start_word,
                    'end_row': word_index - 1,
                    'original_line': line
                })

    # --- helper setters / getters -------------------------------------------------

    def mark_consumed(self, start_row: int, end_row: int, _reason: str, processor_name: str):
        for i in range(start_row, end_row + 1):
            idx = i - 1
            if 0 <= idx < len(self.rows):
                self.rows[idx]['consumed'] = 'YES'
                self.rows[idx]['processor'] = processor_name

    def is_consumed(self, row_number: int) -> bool:
        idx = row_number - 1
        return 0 <= idx < len(self.rows) and self.rows[idx].get('consumed') == 'YES'

    def set_result(self, row_number: int, column: str, value, processor_name: str):
        idx = row_number - 1
        if 0 <= idx < len(self.rows):
            self.rows[idx][column] = value
            self.rows[idx]['processor'] = processor_name
            if self.debug:
                row = self.rows[idx]
                print(f"[TABLE_DEBUG] Row {row_number} after {processor_name}: src='{row.get('source','')}' col='{column}' val='{value}'")

    # --- result reconstruction ----------------------------------------------------
    def _get_final_result_single_line(self) -> str:
        words = []
        for row in self.rows:
            if row.get('consumed') == 'YES' and not any(row.get(col) for col in ['pattern', 'heb2num', 'milon']):
                continue  # Skip rows that were consumed *and* have no replacement text
            if any(row.get(col) and str(row[col]).isdigit() and len(str(row[col])) == 1 for col in ['pattern', 'heb2num', 'milon', 'gender', 'person']):
                continue
            for col in ['pattern', 'heb2num', 'milon']:
                val = row.get(col)
                if val and val not in ('', 'None', None):
                    words.append(f"~{val}~")
                    break
            else:
                words.append(row['source'])
        return ' '.join(words)

    def get_final_result(self) -> str:
        if not self.line_boundaries:
            return self._get_final_result_single_line()
        lines_out = []
        for boundary in self.line_boundaries:
            part_words = []
            for row_num in range(boundary['start_row'], boundary['end_row'] + 1):
                row = self.rows[row_num - 1]
                if row.get('consumed') == 'YES' and not any(row.get(col) for col in ['pattern', 'heb2num', 'milon']):
                    continue  # Skip rows that were consumed *and* have no replacement text
                if any(row.get(col) and str(row[col]).isdigit() and len(str(row[col])) == 1 for col in ['pattern', 'heb2num', 'milon', 'gender']):
                    continue
                for col in ['pattern', 'heb2num', 'milon']:
                    val = row.get(col)
                    if val and val not in ('', 'None', None):
                        part_words.append(f"~{val}~")
                        break
                else:
                    part_words.append(row['source'])
            if part_words:
                lines_out.append(' '.join(part_words))
        return '\n'.join(lines_out)

    # --- simple Excel dump (optional) --------------------------------------------
    def display_table(self, title="Processing Table", line_number=None):
        """Safely print a small textual view of the table for debugging.
        This is a lightweight replacement for the rich HTML console table that
        existed in the original monolithic file.
        """
        if not self.debug:
            return  # Only show when table was created with debug=True

        header = f"[{title}]" if title else "[TABLE]"
        if line_number is not None:
            header += f" line {line_number}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        # Print only first 20 rows to avoid flooding the console
        for row in self.rows[:20]:
            print(row)
        if len(self.rows) > 20:
            print(f"... ({len(self.rows)-20} more rows hidden) ...")

    def export_to_excel(self, filename: str = "hamara_table.xlsx") -> bool:
        try:
            from datetime import datetime
            import openpyxl  # noqa: F401 – ensure dependency hint
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass
            df = pd.DataFrame(self.rows)
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Processing Table')
            print(f"[TABLE] Exported processing table → {filename}")
            return True
        except Exception as exc:
            print(f"[TABLE] Excel export failed: {exc}")
            return False
