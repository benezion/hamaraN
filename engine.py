import abc, os, re, json
from datetime import datetime
import pandas as pd
from consts import *
from table  import TextProcessingTable
from dates  import ConsolidatedDateProcessor
from hebrew import HebrewTextProcessor
import unicodedata 

class aLobe:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def activate(self, **params0):
        pass

class HebrewEnhanceTranslation(aLobe):
    def __init__(self, gender='m2m'):
        self._word_to_row = {}
        self._word_length_map = {}
        self._translation_df = pd.DataFrame({
            'Original': pd.Series(dtype='string'),
            'Nikud': pd.Series(dtype='string'),
            'person_value': pd.Series(dtype='string'),
            'Zachar': pd.Series(dtype='string'),
            'Nekeva': pd.Series(dtype='string')
        })
        self._is_initialized = False
        self.gender = gender
        self.processed_text_cache = {}
        self.processed_numbers = set()
        self.debug = False
        self.debug_file = None
        self.text_processor = HebrewTextProcessor(gender=gender, debug=self.debug)
        self.ssml_input_emotions = None

        # Initialize consolidated date processor (will be set up after loading Hebrew months)
        self.date_processor = None

        self._search_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._hebrew_nouns_gender = {
            'כוכבית': 'f',
            'דקות': 'f',
            'PSI': 'm',
            '%': 'm',  # Percentage (אחוז) is masculine in Hebrew
            'אחוז': 'm',
            'אחוזים': 'm',
        }

        self._hebrew_nouns_types = {}

        self._load_gender_information()

        # Pre-compile date patterns for performance
        self._compile_date_patterns()

        # Initialize consolidated date processor after Hebrew months are available
        self.date_processor = ConsolidatedDateProcessor(
            self.text_processor,
            HEBREW_MONTHS,
            self.heb2num
        )

        self.initialize_dataframe()

    def _unified_number_replace(self, match, track_changes, number_changes, *replace_functions):
        """Unified number replacement method - placeholder implementation"""
        # This method should coordinate different replacement functions
        # For now, return the original match
        return match.group(0)

    def _safe_date_processor(self):
        """Helper method to safely access date_processor with type checking"""
        if not self.date_processor:
            raise RuntimeError("Date processor not initialized")
        return self.date_processor

    def _load_gender_information(self):
        f"""Load gender information from {EXCEL_FILENAME} Noun_Genders sheet"""
        try:
            import pandas as pd
            import os

            possible_paths = [
                EXCEL_FILENAME,
                f'../{EXCEL_FILENAME}',
                f'../../{EXCEL_FILENAME}',
                f'./{EXCEL_FILENAME}'
            ]

            excel_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    excel_path = path
                    break

            if not excel_path:
                raise FileNotFoundError(f"{EXCEL_FILENAME} not found in expected locations")

            print(f"DEBUG: Loading gender information from: {excel_path}")

            df = pd.read_excel(excel_path, sheet_name='Noun_Genders', engine='openpyxl')

            print(f"DEBUG: Excel columns found: {list(df.columns)}")
            print(f"DEBUG: Excel shape: {df.shape}")

            self._hebrew_nouns_gender = {}
            self._hebrew_nouns_types = {}
            self._currency_terms_to_replace = []  # Store currency terms to replace

            for i, row in df.iterrows():
                noun = row.get('שם עצם', '')
                gender = row.get('מין', '')
                noun_type = row.get('סוג', '')

                if pd.notna(noun) and noun.strip():
                    clean_noun = noun.strip()

                    if pd.notna(gender) and gender.strip():
                        gender_value = 'm' if gender.strip() == 'זכר' else 'f'
                        self._hebrew_nouns_gender[clean_noun] = gender_value

                        if clean_noun == 'שנה':
                            print(f"DEBUG: Found שנה in row {i}: gender='{gender}', type='{noun_type}'")

                    if pd.notna(noun_type) and noun_type.strip():
                        self._hebrew_nouns_types[clean_noun] = noun_type.strip()

                        # Extract currency terms where column C (סוג) contains "שקלים"
                        if 'שקלים' in noun_type.strip():
                            self._currency_terms_to_replace.append(clean_noun)

            print(f"Loaded {len(self._hebrew_nouns_gender)} nouns with gender information")
            print(f"Loaded {len(self._hebrew_nouns_types)} nouns with type information")
            print(f"Found {len(self._currency_terms_to_replace)} currency terms to standardize: {self._currency_terms_to_replace}")

            self._hebrew_nouns_gender['שנה'] = 'f'  # שנה is feminine
            self._hebrew_nouns_types['שנה'] = 'תאריך'  # שנה is date-related

        except Exception as e:
            print(f"Warning: Could not load gender information from Excel file: {e}")
            self._hebrew_nouns_gender = {}
            self._hebrew_nouns_types = {}
            self._currency_terms_to_replace = []

            self._hebrew_nouns_gender['שנה'] = 'f'  # שנה is feminine
            self._hebrew_nouns_types['שנה'] = 'תאריך'  # שנה is date-related

    def standardize_currency_terms(self, text):
        """
        Replace all currency terms from column A (שם עצם) where column C (סוג) contains 'שקלים'
        with the standardized word 'שקלים'. This eliminates the need for complex regex patterns
        for various currency representations.
        """
        if not text or not hasattr(self, '_currency_terms_to_replace'):
            return text

        original_text = text

        # Replace all currency terms from the Excel data with 'שקלים'
        # No hardcoded patterns needed - everything is in the Excel file!
        import re
        for currency_term in self._currency_terms_to_replace:
            if currency_term != 'שקלים':  # Don't replace שקלים with itself
                # Use word boundaries to avoid partial replacements
                pattern = r'\b' + re.escape(currency_term) + r'\b'
                text = re.sub(pattern, 'שקלים', text)

        if text != original_text and self.debug:
            print(f"[CURRENCY STANDARDIZATION] '{original_text}' -> '{text}'")

        return text

    def normalize_all_dates_to_dots(self, text):
        """
        Normalize ALL date separators (/, -, _) to dots (.) in a single pass.
        This eliminates the need for multiple date pattern variations and creates
        a single standardized DOT format for all date processing.
        """
        if not text:
            return text

        original_text = text
        import re

        # Convert all date formats to DOT-separated format
        text = re.sub(r'(\d{1,2})\/(\d{1,2})\/(\d{4})', r'\1.\2.\3', text)
        text = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\1.\2.\3', text)
        text = re.sub(r'(\d{1,2})_(\d{1,2})_(\d{4})', r'\1.\2.\3', text)

        # Convert shorter date formats
        text = re.sub(r'(\d{1,2})\/(\d{1,2})(?!/\d)', r'\1.\2', text)
        text = re.sub(r'(\d{1,2})-(\d{1,2})(?!-\d)', r'\1.\2', text)
        text = re.sub(r'(\d{1,2})_(\d{1,2})(?!_\d)', r'\1.\2', text)

        # Convert YYYY formats
        text = re.sub(r'(\d{4})\/(\d{1,2})\/(\d{1,2})', r'\1.\2.\3', text)
        text = re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1.\2.\3', text)
        text = re.sub(r'(\d{4})_(\d{1,2})_(\d{1,2})', r'\1.\2.\3', text)

        text = re.sub(r'(\d{1,2})\/(\d{4})(?!/)', r'\1.\2', text)
        text = re.sub(r'(\d{1,2})-(\d{4})(?!-)', r'\1.\2', text)

        # Handle Hebrew prefix patterns with various separators
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})\/(\d{1,2})(?!/\d)', r'\1\2.\3', text)
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})-(\d{1,2})(?!-\d)', r'\1\2.\3', text)
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})_(\d{1,2})(?!_\d)', r'\1\2.\3', text)

        # Handle Hebrew prefix with full date patterns
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})\/(\d{1,2})\/(\d{4})', r'\1\2.\3.\4', text)
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})-(\d{1,2})-(\d{4})', r'\1\2.\3.\4', text)
        text = re.sub(r'([א-ת]+[-־])(\d{1,2})_(\d{1,2})_(\d{4})', r'\1\2.\3.\4', text)

        if text != original_text and self.debug:
            print(f"[DATE NORMALIZATION] '{original_text}' -> '{text}'")

        return text

    def get_consolidated_date_patterns(self):
        """
        Consolidated date regex patterns - DOT-ONLY after normalization.
        All separators (/, -, _) are converted to dots (.) by normalize_all_dates_to_dots()
        """
        return {
            # Basic dot-separated formats (after normalization)
            'dd_mm_yyyy': r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
            'yyyy_mm_dd': r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
            'dd_mm': r'(?<!\d)(\d{1,2})\.(\d{1,2})(?!\.\d)',
            'mm_yyyy': r'(\d{1,2})\.(\d{4})',

            # Hebrew patterns with month names (don't need normalization)
            'hebrew_prefix_day_month': r'([א-ת]+)-(\d{1,2})\s+ב([א-ת\u0591-\u05C7]+)',
            'hebrew_prefix_day_month_year': r'([א-ת]+)-(\d{1,2})\s+ב([א-ת\u0591-\u05C7]+)\s+(\d{4})',
            'hebrew_month_year': r'(\d+)\s+ב?([א-ת\u0591-\u05C7]+)\s*(\d{4})?',

            # Hebrew prefix with normalized dots
            'hebrew_prefix_dd_mm': r'([א-ת]+[-־])(\d{1,2})\.(\d{1,2})',
            'hebrew_prefix_dd_mm_yyyy': r'([א-ת]+[-־])(\d{1,2})\.(\d{1,2})\.(\d{4})',

            # Special patterns for context words + dates
            'kmo_dd_mm': r'(כמו)\s+(\d{1,2})\.(\d{1,2})',  # כמו 5.4 (like D.M format)
        }

    def _compile_date_patterns(self):
        """Pre-compile date patterns for performance optimization"""
        import re

        # Build Hebrew months pattern once
        hebrew_months_list = list(HEBREW_MONTHS.values())
        months_with_b = [f'ב{month}' for month in hebrew_months_list]
        all_month_patterns = months_with_b + hebrew_months_list
        hebrew_months_pattern = '|'.join(all_month_patterns)

        # Consolidated date patterns - used for both detection and conversion
        self._consolidated_date_patterns = {
            # Date detection patterns (for _is_date_pattern check)
            'detection': [
                re.compile(rf'^תאריך\s+\d+$'),  # "תאריך 9"
                re.compile(rf'^\d+\s+({hebrew_months_pattern})$'),  # "9 באפריל"
                re.compile(rf'^תאריך\s+\d+\s+({hebrew_months_pattern})$'),  # "תאריך 9 באפריל"
            ],

            # Date preprocessing is now handled by normalize_all_dates_to_dots() early in pipeline

            # Date conversion patterns (convert to Hebrew) - Use centralized patterns to avoid duplication!
            'conversion': [
                r'([א-ת])-(\d{2}\.\d{2}\.\d{4})',  # Hebrew-DD.MM.YYYY - MUST BE FIRST (special Hebrew prefix)
                # Use centralized patterns from get_consolidated_date_patterns()
                rf'([א-ת])-(\d+)\s+(ב{hebrew_months_pattern})\s+(\d{{4}})',  # Hebrew-DD בMONTH YYYY
                rf'(\d+)\s+(ב|ל)({hebrew_months_pattern})\s+(\d{{4}})',  # DD [לב]MONTH YYYY
                rf'(\d+)\s+(ב|ל)({hebrew_months_pattern})',  # DD [לב]MONTH (without year)
                rf'(\d+)\s+({hebrew_months_pattern})\s+(\d{{4}})',  # DD MONTH YYYY
                rf'(\d+)\s+({hebrew_months_pattern})',  # DD MONTH (without year)
                rf'({hebrew_months_pattern})\s+(\d{{4}})',  # MONTH YYYY
                # NOTE: Numeric patterns now use centralized get_consolidated_date_patterns()
            ]
        }

        # Legacy compatibility - keep _date_patterns for existing detection code
        self._date_patterns = self._consolidated_date_patterns['detection']

    def _load_phonetic_gender_mappings(self):
        """Load gender mappings for phonetic forms from the main translation dataframe"""
        try:
            if hasattr(self, '_translation_df') and self._translation_df is not None:
                for _, row in self._translation_df.iterrows():
                    original = row.get('Original', '').strip()
                    nikud = row.get('Nikud', '').strip()

                    if original and nikud and original in self._hebrew_nouns_gender:
                        gender = self._hebrew_nouns_gender[original]
                        self._hebrew_nouns_gender[nikud] = gender

                        # No need to create clean nikud mapping - input is already clean
        except Exception as e:
            print(f"Warning: Could not load phonetic gender mappings: {e}")

        self._hebrew_months = HEBREW_MONTHS

        self._hebrew_month_to_num = {v: f'{k:02d}' for k, v in self._hebrew_months.items()}

    def enable_debug_file(self, filename="debug.txt"):
        """Enable debug output to file"""
        if self.debug_file:
            self.debug_file.close()
        self.debug_file = open(filename, 'w', encoding='utf-8')

        # Store the original print function
        import builtins
        self._original_print = builtins.print

        # Override print function to also write to debug file
        def debug_print(*args, **kwargs):
            # Call original print with encoding safety
            try:
                self._original_print(*args, **kwargs)
            except UnicodeEncodeError:
                # Fallback: encode with error handling
                safe_args = []
                for arg in args:
                    if isinstance(arg, str):
                        # Encode to the console's encoding and replace problematic characters
                        safe_arg = arg.encode('cp1255', errors='replace').decode('cp1255')
                        safe_args.append(safe_arg)
                    else:
                        safe_args.append(arg)
                self._original_print(*safe_args, **kwargs)

            # Also write to debug file if it exists
            if self.debug_file and self.debug:
                import io
                output = io.StringIO()
                try:
                    self._original_print(*args, file=output, **kwargs)
                    content = output.getvalue()
                except UnicodeEncodeError:
                    # Fallback for file writing
                    safe_args = []
                    for arg in args:
                        if isinstance(arg, str):
                            # Encode with error handling for file writing
                            safe_arg = arg.encode('cp1255', errors='replace').decode('cp1255')
                            safe_args.append(safe_arg)
                        else:
                            safe_args.append(arg)
                    self._original_print(*safe_args, file=output, **kwargs)
                    content = output.getvalue()
                self.debug_file.write(content)
                self.debug_file.flush()
                output.close()

        # Replace print function
        builtins.print = debug_print

    def _debug_print(self, msg):
        """Print debug message to console and file"""
        if self.debug:
            print(msg)
        # Always write to debug file if it's open, regardless of console debug mode
        if self.debug_file:
            self.debug_file.write(msg + '\n')
            self.debug_file.flush()

    def _debug(self, msg):
        if self.debug:
            print(f"[B_DEBUG] {msg}")
            if self.debug_file:
                self.debug_file.write(f"[B_DEBUG] {msg}\n")
                self.debug_file.flush()



    def sql_initialize_data(self):
        # If already initialized, return
        if self._is_initialized:
            return True

        try:
            # sql_manager: SQLManager = current_app.state.settings["SQL_MANAGER"]
            # data = sql_manager.get_all_from_db(heb_conversion)
            # Note: Database functionality commented out - requires proper imports and configuration

            # new_df = pd.DataFrame([{
            #     'מקור': item.source if item.source != "-" else "",
            #     'מנוקד': item.dotted if item.dotted != "-" else "",
            #     'זכר': item.male if item.male != "-" else "",
            #     'נקבה': item.female if item.female != "-" else "",
            #     'כלל': item.rule if item.rule != "-" else ""
            # } for item in data])

            # Convert SQL data to DataFrame format and replace single quotes with Hebrew geresh
            # new_df = pd.DataFrame([{
            #     'Original': (item.source if item.source != "-" else "").replace("'", "׳"),
            #     'Nikud': (item.dotted if item.dotted != "-" else "").replace("'", "׳"),
            #     'person_value': (item.person_value if item.person_value != "-" else "").replace("'", "׳"),
            #     'Zachar': (item.male if item.male != "-" else "").replace("'", "׳"),
            #     'Nekeva': (item.female if item.female != "-" else "").replace("'", "׳")
            # } for item in data])

            # Database functionality disabled - using Excel file instead
            print("Database functionality is commented out - using Excel file for milon data")
            return True  # Skip database initialization

            # Build word-to-row index for fast lookups
            # new_word_to_row = {self.normalize_double_yud(word): idx for idx, word in enumerate(new_df['Original'])}

            # Build word length map for optimized search
            # word_length_map = defaultdict(set)
            # for word in new_word_to_row:
            #     word_length_map[len(word)].add(word)

            # Store in class variables
            # self._translation_df = new_df
            # self._word_to_row = new_word_to_row
            # self._word_length_map = dict(word_length_map)
            # self._is_initialized = True

            # Load phonetic gender mappings now that the main dataframe is available
            # self._load_phonetic_gender_mappings()

            # print(f"HebrewInhencedTranslotor successfully loaded dictionary with {len(new_df)} entries")
            return True

        except Exception as e:
            print(f"Error loading dictionary: {str(e)}")
            return False

    def normalize_quotes_for_search(self, word):
        """
        Normalize quotes for consistent searching - handles both regular apostrophes and Hebrew geresh
        """
        normalized = word.replace("'", "׳")
        return normalized

    def _is_date_pattern(self, search_word):
        """Check if the search word is a date pattern that should use ordinals instead of milon"""
        search_word_stripped = search_word.strip()

        # Use pre-compiled patterns for much better performance
        for pattern in self._date_patterns:
            if pattern.match(search_word_stripped):
                return True

        return False

    def _should_skip_dictionary_lookup(self, word):
        """
        Conservative early exit strategy - only skip obvious non-dictionary patterns
        BUT allow dictionary lookup for numbers that might be part of phrases like "מוקד XXX"
        """
        # REMOVED: if word.isdigit(): return True
        # Allow pure numbers to be checked in dictionary for phrases like "מוקד XXX"

        if re.match(r'^\d+\.\d+$', word):
            return True

        # Skip ו and ל prefixed numbers (no special milon lookup)
        # But allow ב- prefixed numbers to be checked in milon first
        if re.match(r'^[ול]-?\d+(\.\d+)?$', word):
            return True

        if re.match(r'^[12]\d{3}$', word):
            return True

        return False

    def _search_raw_milon(self, search_word):
        """
        OPTIMIZED: Search for a word in the appropriate milon table based on word count
        Returns raw DataFrame
        """
        if self._should_skip_dictionary_lookup(search_word):
            return pd.DataFrame()

        if search_word in self._search_cache:
            self._cache_hits += 1
            return self._search_cache[search_word]

        self._cache_misses += 1

        if self._is_date_pattern(search_word):
            result = pd.DataFrame()
            self._search_cache[search_word] = result
            return result

        normalized_search = self._TextCleaningUtils.normalize_quotes(search_word)

        # Use unified table for all searches - much faster than separate tables
        search_table = self._milon_unified if hasattr(self, '_milon_unified') else self._translation_df

        # Try exact match first
        result = self._search_sorted_table(search_table, normalized_search)

        # If not found, try stripping common prefixes
        if result.empty:
            result = self._search_with_prefix_stripping(search_table, normalized_search)

        self._search_cache[search_word] = result
        return result

    def _search_with_prefix_stripping(self, search_table, normalized_search):
        """
        Try searching with common Hebrew prefixes stripped
        Returns DataFrame result, potentially with prefix information
        """
        # Common Hebrew prefixes to try stripping (order matters - longer first)
        prefixes_to_strip = ['כש', 'ש', 'ו', 'ה', 'ב', 'ל', 'מ', 'כ']

        words = normalized_search.split()
        if not words:
            return pd.DataFrame()

        # Try stripping prefix from first word only (most common case)
        first_word = words[0]

        for prefix in prefixes_to_strip:
            if first_word.startswith(prefix) and len(first_word) > len(prefix):
                stripped_first = first_word[len(prefix):]

                # Reconstruct search phrase with stripped first word
                stripped_words = [stripped_first] + words[1:]
                stripped_search = ' '.join(stripped_words)

                # Search for the stripped version
                result = self._search_sorted_table(search_table, stripped_search)

                if not result.empty:
                    # Found a match! Store the prefix info for later reconstruction
                    # We'll modify the result to include prefix information
                    result_copy = result.copy()

                    # Add prefix info to the result for later reconstruction
                    if 'Original' in result_copy.columns:
                        original_val = result_copy.iloc[0]['Original']
                        # Store original search word to maintain prefix in final output
                        col_loc = result_copy.columns.get_loc('Original')
                        result_copy.iloc[0, col_loc] = normalized_search  # type: ignore

                        # Store the stripped match in a new column for processing
                        if 'Stripped_Match' not in result_copy.columns:
                            result_copy['Stripped_Match'] = ''
                        col_loc = result_copy.columns.get_loc('Stripped_Match')
                        result_copy.iloc[0, col_loc] = original_val  # type: ignore

                        # Store the prefix for reconstruction
                        if 'Prefix' not in result_copy.columns:
                            result_copy['Prefix'] = ''
                        col_loc = result_copy.columns.get_loc('Prefix')
                        result_copy.iloc[0, col_loc] = prefix  # type: ignore

                    if self.debug:
                        self._debug(f"Found match with prefix stripping: '{normalized_search}' -> stripped '{stripped_search}' -> found match")

                    return result_copy

        # No match found even with prefix stripping
        return pd.DataFrame()

    def search_in_milon(self, search_word, person_context, source_gender, target_gender,
                       source_tts_gender, target_tts_gender, gender_context,
                       existing_highlight_blocks, text, leading_punct="", trailing_punct=""):
        """
        Search milon and return processed, marked text ready to use
        Returns: (marked_text, raw_text, tts_gender, found)
        """
        row = self._search_raw_milon(search_word)
        if len(row) == 0:  # More efficient than .empty
            return None, None, None, False, False

        # CRITICAL FIX: Check for person indicators before using gender columns
        has_person_indicator = (
            self.text_processor.global_context.get('detected_gender') is not None or
            self.text_processor.global_context.get('second_person_detected') or
            self.text_processor.global_context.get('force_maintain_gender') or
            person_context in ["FIRST_PERSON", "SECOND_PERSON"]
        )

        if self.debug:
            print(f"[DEBUG] search_in_milon: Person indicator detected: {has_person_indicator}")

        # Track whether we used the original word as fallback
        used_original_word = False
        
        # Process replacement using existing logic
        if 'Nikud' in row.columns and pd.notna(row.iloc[0]['Nikud']) and str(row.iloc[0]['Nikud']).strip() != '':
            if has_person_indicator:
                # Person indicators found - use full gender logic
                replacement, tts_gender = self.text_processor.process_nikud_replacement(
                    row, person_context, False, source_gender, target_gender,
                    source_tts_gender, target_tts_gender, gender_context, self.debug
                )
            else:
                # No person indicators - ONLY use Nikud column, ignore gender columns
                replacement = str(row.iloc[0]['Nikud']).strip()
                tts_gender = "male"  # Default TTS gender when no person context
                if self.debug:
                    print(f"[DEBUG] No person indicators - using ONLY Nikud column: '{replacement}'")
        else:
            if has_person_indicator:
                # Handle cases where Nikud column is empty - use gender columns
                person_value = row.iloc[0].get('person_value') if 'person_value' in row.columns else None
                column_to_use, tts_gender = self.text_processor.determine_column_and_gender(
                    person_context, False, source_gender, target_gender, gender_context, True, False, person_value
                )
                replacement = row.iloc[0][column_to_use]
            else:
                # No person indicators AND no Nikud - check if it's a slash form
                if '/' in search_word:
                    # Slash form without person indicators - show both forms: "זכר, נקבה"

                    zachar_val = row.iloc[0].get('Zachar', '') if 'Zachar' in row.columns else ''
                    nekeva_val = row.iloc[0].get('Nekeva', '') if 'Nekeva' in row.columns else ''
                    
                    if pd.notna(zachar_val) and pd.notna(nekeva_val) and str(zachar_val).strip() and str(nekeva_val).strip():
                        replacement = f"{str(zachar_val).strip()}, {str(nekeva_val).strip()}"
                        tts_gender = "male"  # Default TTS gender for combined forms
                        if self.debug:
                            print(f"[DEBUG] Slash form '{search_word}' with no person indicators - combining both forms: '{replacement}'")
                    else:
                        # Fallback to original if gender columns are empty
                        replacement = row.iloc[0]['Original']
                        used_original_word = True
                        tts_gender = "male"
                        if self.debug:
                            print(f"[DEBUG] Slash form '{search_word}' - gender columns empty, using original: '{replacement}'")
                else:
                    # Regular word - use original word
                    replacement = row.iloc[0]['Original']
                    used_original_word = True
                    tts_gender = "male"  # Default TTS gender
                    if self.debug:
                        print(f"[DEBUG] No person indicators and no Nikud - using original word: '{replacement}'")

            if pd.isna(replacement) or not str(replacement).strip():
                # If chosen gender column is empty, check if both Zachar and Nekeva are empty
                zachar_val = row.iloc[0].get('Zachar', '')
                nekeva_val = row.iloc[0].get('Nekeva', '')

                # If both gender columns are empty, use column B (backtick column) regardless of gender
                if (pd.isna(zachar_val) or not str(zachar_val).strip()) and \
                   (pd.isna(nekeva_val) or not str(nekeva_val).strip()):
                    backtick_val = row.iloc[0].get('`', '')
                    if not pd.isna(backtick_val) and str(backtick_val).strip():
                        replacement = str(backtick_val).strip()
                        if self.debug:
                            self._debug(f"Both gender columns empty, using backtick column: '{replacement}'")
                    else:
                        # PRONOUN FALLBACK DISABLED: Now using explicit milon entries with אַתְּ
                        # The milon contains proper את + verb → אַתְּ + verb entries
                        # If not found in milon, keep original word (likely direct object marker)
                        replacement = row.iloc[0]['Original']
                        used_original_word = True
                        if self.debug:
                            self._debug(f"No conversion data found, using original word: '{replacement}'")
                else:
                    replacement = row.iloc[0]['Original']
                    # Mark that we used original word so we don't double-add prefix
                    used_original_word = True
                    if self.debug:
                        self._debug(f"Column '{column_to_use}' is empty, using original word: '{replacement}'")

        # Handle prefix reconstruction if this was found via prefix stripping
        if 'Prefix' in row.columns and pd.notna(row.iloc[0]['Prefix']) and str(row.iloc[0]['Prefix']).strip():
            prefix = str(row.iloc[0]['Prefix']).strip()
            if replacement and not used_original_word:
                # Add prefix back only if we didn't use original word (which already has prefix)
                original_replacement = replacement
                replacement = prefix + replacement
                if self.debug:
                    self._debug(f"Reconstructed with prefix: '{prefix}' + '{original_replacement}' -> '{replacement}'")
            elif self.debug and used_original_word:
                self._debug(f"Skipping prefix reconstruction - used original word: '{replacement}'")

        # Handle gender tracking and check if exclusive gender was found
        is_exclusive_gender = self._update_milon_gender_tracking(row, search_word)

        # Always add  markers for milon results (simpler approach)
        marked_replacement = replacement

        # Add punctuation if provided
        if leading_punct or trailing_punct:
            marked_replacement = f'{leading_punct}{marked_replacement}{trailing_punct}'

        return marked_replacement, replacement, tts_gender, True, is_exclusive_gender

    def get_html_highlighted_result(self, table):
        """
        Generate HTML result with ~...~ markers based on table processing.
        Any text that comes from processors (not from original column A) gets ~...~ markers.
        """
        result_words = []

        for row in table.rows:
            final_word = None
            needs_highlighting = False

            # Priority order: milon > heb2num > pattern > source
            if row.get('milon'):
                final_word = row['milon']
                needs_highlighting = True  # Milon conversion gets highlighted
            elif row.get('heb2num'):
                final_word = row['heb2num']
                needs_highlighting = True  # Number conversion gets highlighted
            elif row.get('pattern'):
                final_word = row['pattern']
                needs_highlighting = True  # Pattern conversion gets highlighted
            elif not row.get('consumed'):  # Only include if not consumed
                final_word = row['source']
                needs_highlighting = False  # Original text doesn't get highlighted

            if final_word:
                if needs_highlighting:
                    result_words.append(f"~{final_word}~")
                else:
                    result_words.append(final_word)

        html_result = ' '.join(result_words)
        return html_result

    def get_table_dictionary_changes(self, table):
        """Extract dictionary changes from table for compatibility with display system"""
        dict_words = {}
        for row in table.rows:
            if row.get('milon'):
                dict_words[row['source']] = row['milon']
            elif row.get('heb2num'):
                dict_words[row['source']] = row['heb2num']
            elif row.get('pattern'):
                dict_words[row['source']] = row['pattern']
        return dict_words

    def _add_terminal_colors(self, text):
        """Convert ~...~ markers to red terminal colors (ANSI codes)"""
        if not text:
            return text

        # ANSI color codes
        RED = '\033[91m'
        RESET = '\033[0m'

        # Replace ~content~ with red colored content
        def color_match(match):
            content = match.group(1)  # Content between ~ markers
            return f'{RED}{content}{RESET}'

        # Apply red color to content between ~ markers and remove the markers
        colored_text = re.sub(r'~([^~]+)~', color_match, text)

        return colored_text

    def process_text_with_table(self, text, line_number=None):
        """NEW: Process text using table-based architecture with ~ markers for HTML"""
        import time
        start_time = time.perf_counter()

        # Reset global context and caches for each new line to ensure clean processing
        if hasattr(self, 'text_processor'):
            self.text_processor.reset_global_context(debug_flag=self.debug)

        if not text:
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            if self.debug:
                print(f"[RUNTIME] PROCESS_TEXT_WITH_TABLE RUNTIME: {runtime_ms:.3f}ms (empty input)")
            return {
                'ssml_text': '',
                'ssml_clean': '',
                'ssml_marked': '',
                'show_text': '',
                'dict_words': {},
                'original_input': ''
            }

        # EARLY TEXT NORMALIZATION
        import re
        original_text = text

        # Replace Hebrew maqaf (־) with regular hyphen (-)
        text = text.replace('־', '-')

        # Replace em dash (—) with space
        text = text.replace('—', ' ')

        # STANDARDIZE CURRENCY TERMS: Replace all currency variations with "שקלים"
        text = self.standardize_currency_terms(text)

        # NORMALIZE DATE FORMATS: Convert all date separators (/, -, _) to dots (.)
        text = self.normalize_all_dates_to_dots(text)

        # NORMALIZE TIME RANGES: Fix dot format errors (HH:MM.HH:MM → HH:MM עד HH:MM)
        # Must come AFTER date normalization to avoid conflicts
        text = re.sub(r'(\d{1,2}:\d{2})\.(\d{1,2}:\d{2})', r'\1 עד \2', text)

        if self.debug:
            print(f"[PROCESS] NEW TABLE-BASED processing for: '{text}'")

        # Ensure optimized tables are loaded (especially when debug mode is enabled after construction)
        # SMART CACHING: Only reload if tables are missing or debug mode changed
        need_reload = (
            not hasattr(self, '_milon_dict') or
            not hasattr(self, '_milon_unified') or
            len(getattr(self, '_milon_dict', {})) == 0
        )

        if need_reload:
            if self.debug:
                print(f"[INIT_DEBUG] Loading milon data (not cached)...")
            self.initialize_dataframe()
        elif self.debug:
            print(f"[INIT_DEBUG] Using cached milon data ({len(self._milon_dict)} entries)")


        # Create table
        table = self.create_processing_table(text)

        # Pre-populate gender information from milon/noun libraries
        self.populate_gender_information(table)

        # Make table available for early processing capture
        self.current_processing_table = table

        # Run processors (silently)
        # IMPORTANT: Milon (dictionary) ALWAYS comes first - highest priority
        self.run_milon_processor(table)
        # Then run number processing on original source text for patterns not in milon
        self.run_complete_number_processor_on_source(table)
        self.run_individual_number_processor(table)  # Ensure individual numbers are processed
        self.run_pattern_processor(table)

        # Get results
        ssml_marked = table.get_final_result()  # This has ~ markers for HTML highlighting
        ssml_clean = re.sub(r'~([^~]+)~', r'\1', ssml_marked).replace('~', '')  # Clean version for TTS
        dict_words = self.get_table_dictionary_changes(table)

        # Store table for debugging
        self.last_processing_table = table

        # Create full SSML with <speak> tags for TTS
        clean_hebrew_text = ssml_clean
        emotion_tag = self.ssml_input_emotions if self.ssml_input_emotions else None

        if emotion_tag:
            final_ssml = f"<speak>\n{original_text}\n<say-as>\n<{emotion_tag}>{clean_hebrew_text}</>\n</say-as>\n</speak>"
        else:
            final_ssml = f"<speak>\n{original_text}\n<say-as>\n{clean_hebrew_text}\n</say-as>\n</speak>"

        # END TIMING HERE - before Excel export and debug output
        end_time = time.perf_counter()
        runtime_ms = (end_time - start_time) * 1000
        if self.debug:
            print(f"[RUNTIME] PROCESS_TEXT_WITH_TABLE RUNTIME: {runtime_ms:.3f}ms")

        # Print debug table only once at the end (after timing)
        if self.debug:
            table.display_table("[FINAL] PROCESSING TABLE", line_number)
            self._print_search_statistics()

        if self.debug:
            try:
                print(f"  Clean result: '{ssml_clean}'")
                print(f"  Marked result: '{ssml_marked}'")
                print(f"  Dictionary changes: {dict_words}")
                print(f"  Red colored result: '{self._add_terminal_colors(ssml_marked)}'")
                print(f"[SSML] FULL SSML BUFFER (with <speak> tags):")
                print(final_ssml)
            except UnicodeEncodeError:
                print("  [DEBUG] Unicode output suppressed due to console encoding limitations")
                print(f"  Dictionary changes count: {len(dict_words) if dict_words else 0}")
                print(f"  SSML text length: {len(final_ssml)} characters")

        return {
            'ssml_text': final_ssml,
            'ssml_clean': ssml_clean,
            'ssml_marked': ssml_marked,
            'show_text': text,  # Original input for display
            'dict_words': dict_words,
            'original_input': text,
            'processing_table': table  # Include table for export functionality
        }

    def run_complete_number_processor_on_source(self, table):
        """Run COMPLETE number processing on SOURCE TEXT and store ALL results immediately in table"""
        if self.debug:
            print("[NUMBER] Running COMPLETE NumberProcessor on SOURCE TEXT...")

        # Step 1: Get original source text from table
        original_text = ' '.join([row['source'] for row in table.rows])

        if self.debug:
            print(f"  Source text for number processing: '{original_text}'")

        # Step 2: Run COMPLETE number processing from the old algorithm
        processed_text, number_changes = self.process_numbers_with_noun_context(original_text, track_changes=True, table=table)

        if self.debug:
            print(f"  Number processing result: '{processed_text}'")
            print(f"  Number changes: {number_changes}")

        # Step 3: Store ALL number conversion results immediately in the table
        # This includes decimal currency, regular numbers, dates, etc.
        self._store_all_number_changes_in_source_table(table, original_text, number_changes)



    def _store_conversion_in_table(self, table, original_key, converted_value, processor_name="DirectConversion"):
        """Store a single conversion result directly in the table"""
        if not table or not original_key or not converted_value:
            return False

        # Handle single word conversions
        if len(original_key.split()) == 1:
            # Try exact match first
            for i, row in enumerate(table.rows):
                if row['source'] == original_key:
                    # Check if milon column has content (milon always takes priority)
                    existing_milon = row.get('milon', '')
                    if existing_milon and existing_milon not in ['None', '']:
                        if self.debug:
                            print(f"    ⚠️  Skipping direct match: row {i + 1} '{original_key}' already has milon = '{existing_milon}' (milon priority)")
                        continue  # Continue looking for other matching rows

                    # Check if heb2num column is empty or contains only 'None' before storing
                    existing_heb2num = row.get('heb2num', '')
                    if not existing_heb2num or existing_heb2num in ['None', '']:
                        table.set_result(i + 1, 'heb2num', converted_value, processor_name)
                        if self.debug:
                            print(f"    ✅ Direct table storage: row {i + 1} '{original_key}' = '{converted_value}'")
                        return True
                    else:
                        if self.debug:
                            print(f"    ⚠️  Skipping direct match: row {i + 1} '{original_key}' already has heb2num = '{existing_heb2num}'")
                        continue  # Continue looking for other matching rows

            # Try normalized match (for dates like ל-2025-01-03 -> 2025-01-03)
            # ALSO handle date sub-patterns (like "07" within "07.03.2025")
            for i, row in enumerate(table.rows):
                source_text = row['source']
                # Normalize both source and key to handle dot/hyphen differences
                normalized_source = source_text.replace('.', '-')
                normalized_key = original_key.replace('.', '-')

                # Check for exact substring match (dates) or exact match
                is_date_subpattern = (original_key.isdigit() and
                                    len(original_key) <= 4 and
                                    original_key in source_text and
                                    ('.' in source_text or '/' in source_text))

                if (normalized_key in normalized_source and len(original_key) > 4) or is_date_subpattern:
                    # Check if milon column has content (milon always takes priority)
                    existing_milon = row.get('milon', '')
                    if existing_milon and existing_milon not in ['None', '']:
                        if self.debug:
                            print(f"    ⚠️  Skipping normalized match: '{original_key}' found in '{source_text}' row {i + 1} - already has milon = '{existing_milon}' (milon priority)")
                        continue  # Continue looking for other matching rows

                    # Check if heb2num column is empty before storing
                    existing_heb2num = row.get('heb2num', '')
                    if not existing_heb2num or existing_heb2num in ['None', '']:
                        # For PREFIXED_NUMBER, the converted_value already contains the correct prefix, so use as-is
                        if processor_name == "PREFIXED_NUMBER":
                            table.set_result(i + 1, 'heb2num', converted_value, processor_name)
                            if self.debug:
                                print(f"    ✅ Normalized table storage (prefixed number): '{original_key}' found in '{source_text}' row {i + 1} = '{converted_value}'")
                        # For date sub-patterns, we need to convert the entire date, not just the sub-part
                        elif is_date_subpattern:
                            # Try to convert the full date pattern
                            full_date_conversion = self._try_convert_full_date_pattern(source_text, original_key, converted_value)
                            if full_date_conversion:
                                table.set_result(i + 1, 'heb2num', full_date_conversion, processor_name)
                                if self.debug:
                                    print(f"    ✅ Date pattern table storage: '{source_text}' row {i + 1} = '{full_date_conversion}' (from sub-pattern '{original_key}')")
                            else:
                                # Fallback: just store the sub-pattern conversion
                                table.set_result(i + 1, 'heb2num', converted_value, processor_name)
                                if self.debug:
                                    print(f"    ✅ Date sub-pattern table storage: '{original_key}' in '{source_text}' row {i + 1} = '{converted_value}'")
                        # For date patterns with prefix, preserve the prefix in the conversion
                        elif '-' in source_text and not original_key.startswith(source_text.split('-')[0]):
                            # Extract prefix from source (e.g., "ל" from "ל-2025-01-03")
                            prefix = source_text.split('-')[0]
                            # Add prefix to conversion (e.g., "ל" + "שלישי בינואר..." = "לשלישי בינואר...")
                            prefixed_conversion = f"{prefix}{converted_value}"
                            table.set_result(i + 1, 'heb2num', prefixed_conversion, processor_name)
                            if self.debug:
                                print(f"    ✅ Normalized table storage with prefix: '{original_key}' found in '{source_text}' row {i + 1} = '{prefixed_conversion}'")
                        else:
                            table.set_result(i + 1, 'heb2num', converted_value, processor_name)
                            if self.debug:
                                print(f"    ✅ Normalized table storage: '{original_key}' found in '{source_text}' row {i + 1} = '{converted_value}'")
                        return True
                    else:
                        if self.debug:
                            print(f"    ⚠️  Skipping normalized match: '{original_key}' found in '{source_text}' row {i + 1} - already has heb2num = '{existing_heb2num}'")
                        continue  # Continue looking for other matching rows
        else:
            # Handle multi-word conversions (like "חודש ינואר 2025")
            original_words = original_key.split()
            first_word = original_words[0]

            # Currency patterns are now standardized to שקלים early in pipeline
            def try_sequence_match(words_to_match):
                for start_i, row in enumerate(table.rows):
                    if row['source'] == words_to_match[0]:
                        # Check if this is the start of our target sequence
                        sequence_matches = True
                        for j, target_word in enumerate(words_to_match):
                            check_row_idx = start_i + j
                            if (check_row_idx >= len(table.rows) or
                                table.rows[check_row_idx]['source'] != target_word):
                                sequence_matches = False
                                break

                        if sequence_matches:
                            return start_i
                return None

            # Try original sequence first
            start_i = try_sequence_match(original_words)

            # Currency format alternatives no longer needed - everything is standardized to שקלים early in pipeline

            if start_i is not None:
                # Check if any word in the sequence is already modified
                sequence_already_modified = False
                for j in range(len(original_words)):
                    check_row_idx = start_i + j
                    if (check_row_idx < len(table.rows) and
                        table.rows[check_row_idx].get('heb2num')):
                        sequence_already_modified = True
                        if self.debug:
                            print(f"    ⚠️  Skipping multi-word sequence starting at row {start_i + 1} - word '{table.rows[check_row_idx]['source']}' already modified")
                        break

                if not sequence_already_modified:
                    # Found the correct sequence! Store the conversion
                    table.set_result(start_i + 1, 'heb2num', converted_value, processor_name)
                    if self.debug:
                        print(f"    ✅ Multi-word table storage: row {start_i + 1} = '{converted_value}'")

                    # Mark subsequent words with span markers
                    span_size = len(original_words)
                    for j in range(1, span_size):
                        span_row_idx = start_i + j
                        if span_row_idx < len(table.rows):
                            table.set_result(span_row_idx + 1, 'heb2num', str(span_size), processor_name)
                            if self.debug:
                                print(f"    ✅ Multi-word span marker: row {span_row_idx + 1} = '{span_size}'")

                    return True

        if self.debug:
            print(f"    ❌ Could not store '{original_key}' in table")
        return False

    def _try_convert_full_date_pattern(self, source_text, original_key, converted_value):
        """Try to convert a full date pattern when we have a sub-pattern conversion"""
        import re

        # Handle DD.MM.YYYY pattern
        dd_mm_yyyy_match = re.match(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', source_text)
        if dd_mm_yyyy_match:
            day_str, month_str, year_str = dd_mm_yyyy_match.groups()

            try:
                day_num = int(day_str)
                month_num = int(month_str)
                year_num = int(year_str)

                if 1 <= day_num <= 31 and 1 <= month_num <= 12 and 1900 <= year_num <= 2100:
                    # Convert each part to Hebrew
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    # Format as Hebrew date
                    return self._safe_date_processor().format_hebrew_date(day_text, month_name, year_text)

            except (ValueError, AttributeError):
                pass

        # Handle MM.YYYY pattern
        mm_yyyy_match = re.match(r'(\d{1,2})\.(\d{4})', source_text)
        if mm_yyyy_match:
            month_str, year_str = mm_yyyy_match.groups()

            try:
                month_num = int(month_str)
                year_num = int(year_str)

                if 1 <= month_num <= 12 and 1900 <= year_num <= 2100:
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                    return f"{month_name} {year_text}"

            except (ValueError, AttributeError):
                pass

        return None  # Could not convert

    def _store_all_number_changes_in_source_table(self, table, original_text, number_changes):
        """Store ALL number conversion results from source text processing into the table"""
        if not number_changes:
            if self.debug:
                print("  No number changes to store from source processing")
            return

        if self.debug:
            print(f"  Storing {len(number_changes)} number changes from source processing...")

        for original_phrase, converted_phrase in number_changes.items():
            if self.debug:
                print(f"    Processing: '{original_phrase}' → '{converted_phrase}'")

            # Handle different types of number conversions
            if original_phrase.isdigit():
                # Single number conversion (like "2500" → "אַלְפַּיִם חֲמֵשׁ מֵאוֹת")
                for i, row in enumerate(table.rows):
                    if row['source'] == original_phrase:
                        # Check if this row already has a heb2num value - don't overwrite direct table storage
                        existing_value = row.get('heb2num')
                        if existing_value and existing_value not in ['None', '']:
                            if self.debug:
                                print(f"    [SKIP] Skipping '{original_phrase}' (row {i + 1}) - already has value '{existing_value}'")
                        else:
                            table.set_result(i + 1, 'heb2num', converted_phrase, 'SourceNumberProcessor')
                            if self.debug:
                                print(f"    ✅ Stored single number: row {i + 1} = '{converted_phrase}'")
                        break

            elif len(original_phrase.split()) == 1:
                # Single token with numbers (like "8%" → "שְׁמוֹנָה אחוזים", "5.5%" → Hebrew)
                stored = False
                for i, row in enumerate(table.rows):
                    if row['source'] == original_phrase:
                        table.set_result(i + 1, 'heb2num', converted_phrase, 'SourceNumberProcessor')
                        if self.debug:
                            print(f"    ✅ Stored single token: row {i + 1} = '{converted_phrase}'")
                        stored = True
                        break

                # Handle cases where date normalization changes format (e.g., "5-4-2025" from "ה-5.4.2025")
                if not stored:
                    for i, row in enumerate(table.rows):
                        # Check if normalized source contains the conversion key
                        normalized_source = row['source'].replace('.', '-')
                        # For date patterns like "ה-5.4.2025" -> "ה-5-4-2025", check if key "5-4-2025" is in normalized source
                        if original_phrase in normalized_source and len(original_phrase) > 4:
                            # Check if the row already has a prefixed conversion - don't overwrite it
                            existing_conversion = row.get('heb2num', '')
                            if existing_conversion and existing_conversion != 'None' and existing_conversion.startswith(row['source'].split('-')[0]):
                                if self.debug:
                                    print(f"    ⚠️  Skipping normalized match: '{original_phrase}' - row {i + 1} already has prefixed conversion: '{existing_conversion}'")
                                stored = True
                                break

                            # Check if heb2num column is empty before storing
                            existing_heb2num = row.get('heb2num', '')
                            if existing_heb2num and existing_heb2num not in ['None', '']:
                                if self.debug:
                                    print(f"    ⚠️  Skipping normalized match: '{original_phrase}' - row {i + 1} already has heb2num = '{existing_heb2num}'")
                                continue  # Continue looking for other matching rows

                            # For date patterns with prefix, preserve the prefix in the conversion
                            if '-' in row['source'] and not original_phrase.startswith(row['source'].split('-')[0]):
                                # Extract prefix from source (e.g., "מ" from "מ-2025-03-07")
                                prefix = row['source'].split('-')[0]
                                # Add prefix to conversion (e.g., "מ" + "שביעי במרץ..." = "משביעי במרץ...")
                                prefixed_conversion = f"{prefix}{converted_phrase}"
                                table.set_result(i + 1, 'heb2num', prefixed_conversion, 'SourceNumberProcessor')
                                if self.debug:
                                    print(f"    ✅ Stored normalized match with prefix: '{original_phrase}' found in '{row['source']}' row {i + 1} = '{prefixed_conversion}'")
                            else:
                                table.set_result(i + 1, 'heb2num', converted_phrase, 'SourceNumberProcessor')
                                if self.debug:
                                    print(f"    ✅ Stored normalized match: '{original_phrase}' found in normalized '{normalized_source}' (original: '{row['source']}') row {i + 1} = '{converted_phrase}'")
                            stored = True
                            break

            elif len(original_phrase.split()) >= 2:
                # Multi-word conversion (like "ל-810.27 שקלים" → Hebrew or "חודש מרץ 2025" → Hebrew)
                original_words = original_phrase.split()

                # Find the CORRECT SEQUENCE of words in the table (not just the first occurrence)
                first_word = original_words[0]
                found_sequence = False

                for start_i, row in enumerate(table.rows):
                    if row['source'] == first_word:
                        # Check if this is the start of our target sequence
                        sequence_matches = True
                        for j, target_word in enumerate(original_words):
                            check_row_idx = start_i + j
                            if (check_row_idx >= len(table.rows) or
                                table.rows[check_row_idx]['source'] != target_word):
                                sequence_matches = False
                                break

                        if sequence_matches:
                            # Check if any word in the sequence is already modified
                            sequence_already_modified = False
                            for j in range(len(original_words)):
                                check_row_idx = start_i + j
                                if (check_row_idx < len(table.rows) and
                                    table.rows[check_row_idx].get('heb2num')):
                                    sequence_already_modified = True
                                    if self.debug:
                                        print(f"    ⚠️  Skipping sequence starting at row {start_i + 1} - word '{table.rows[check_row_idx]['source']}' already modified")
                                    break

                            if not sequence_already_modified:
                                # Found the correct sequence! Store the conversion
                                table.set_result(start_i + 1, 'heb2num', converted_phrase, 'SourceNumberProcessor')
                                if self.debug:
                                    print(f"    ✅ Stored multi-word conversion: row {start_i + 1} = '{converted_phrase}'")

                                # Mark subsequent words with span markers
                                span_size = len(original_words)
                                for j in range(1, span_size):
                                    span_row_idx = start_i + j
                                    if span_row_idx < len(table.rows):
                                        table.set_result(span_row_idx + 1, 'heb2num', str(span_size), 'SourceNumberProcessor')
                                        if self.debug:
                                            print(f"    ✅ Stored span marker: row {span_row_idx + 1} = '{span_size}'")

                                found_sequence = True
                                break

    def store_processing_table(self, table):
        """Store the processing table for HTML generation"""
        self.last_processing_table = table

    def create_processing_table(self, text):
        """Create a new processing table"""
        return TextProcessingTable(text, debug=self.debug)

    def populate_gender_information(self, table):
        """Pre-populate gender and person information from milon/noun libraries and person context"""
        if not table or not table.rows:
            return

        if self.debug:
            print(f"[GENDER] Pre-populating gender and person information for {len(table.rows)} words...")

        # PHASE 1: Pre-scan to detect person indicators and set global context early
        for i, row in enumerate(table.rows):
            word = row['source']
            if not word or word.isdigit():
                continue

            clean_word = self._clean_word_for_gender_lookup(word)
            if not clean_word:
                continue

            # Just check for person indicators to set global context
            # Pass text context for direct object marker detection
            original_text = table.original_text if hasattr(table, 'original_text') else ""
            word_position = self._calculate_word_position(original_text, row['source'], i)
            person_context, is_first_person, is_second_person, inherent_gender, is_verb_form, is_neutral_second_person = self.text_processor.check_person_indicators(clean_word, None, original_text, word_position)

            # CRITICAL: Also call determine_inherent_gender to set global context from pronouns
            self.text_processor.determine_inherent_gender(clean_word, original_text, word_position, update_global=True)
            # Global context is now set if any person indicators were found

        if self.debug and self.text_processor.global_context['detected_gender']:
            print(f"  [PRE-SCAN] Global gender context detected: {self.text_processor.global_context['detected_gender']}")

        current_person_gender = None  # Track the current active person context gender

        for i, row in enumerate(table.rows):
            word = row['source']

            # Skip empty words or span markers
            if not word or word.isdigit():
                continue

            # Clean the word for lookup (remove punctuation, prefixes)
            clean_word = self._clean_word_for_gender_lookup(word)
            if not clean_word:
                continue

            # DEBUG: Show word cleaning results
            if self.debug and word == 'את':
                print(f"  [DEBUG] Word cleaning: '{word}' → '{clean_word}'")

            # CHECK PERSON INDICATORS AND SET PERSON + GENDER CONTEXT
            # Pass text context for direct object marker detection
            original_text = table.original_text if hasattr(table, 'original_text') else ""
            word_position = self._calculate_word_position(original_text, word, i)
            person_context, is_first_person, is_second_person, inherent_gender, is_verb_form, is_neutral_second_person = self.text_processor.check_person_indicators(clean_word, None, original_text, word_position)

            # Determine person context and update current gender context
            person_value = None
            if is_first_person:
                person_value = "1"
                table.set_result(i + 1, 'person', person_value, 'PersonPreprocessor')
                # Update current active person gender context
                # First person = speaker gender (first character)
                source_gender = self.gender[0] if len(self.gender) >= 1 else 'f'
                current_person_gender = source_gender

            elif is_second_person:
                person_value = "2"
                table.set_result(i + 1, 'person', person_value, 'PersonPreprocessor')
                # Update current active person gender context
                # Second person = listener gender (third character)
                target_gender = self.gender[2] if len(self.gender) >= 3 else self.gender[0]
                current_person_gender = target_gender

            # Determine which gender form to use
            gender_to_use = None
            if person_value:  # If this word is a person indicator
                gender_to_use = current_person_gender
                if gender_to_use:
                    table.set_result(i + 1, 'gender', gender_to_use, 'PersonPreprocessor')
                    if self.debug:
                        print(f"  ✅ Row {i + 1}: '{word}' → person: {person_value}, gender: {gender_to_use}")
            else:
                # Not a person indicator - check if we should use current person context
                # Calculate word position in original text for context-aware gender detection
                original_text = table.original_text if hasattr(table, 'original_text') else ""
                word_position = self._calculate_word_position(original_text, word, i)
                inherent_gender = self._lookup_gender_from_sources(clean_word, word, original_text, word_position)

                # Check if there's a global gender context from detected pronouns
                global_gender = None
                if self.text_processor.global_context['detected_gender']:
                    global_gender = self.text_processor.global_context['detected_gender']
                    global_gender_letter = 'm' if global_gender == 'male' else 'f'

                if inherent_gender and current_person_gender:
                    # This word has inherent gender and we have active person context
                    # Use the current person context gender for agreement
                    table.set_result(i + 1, 'gender', current_person_gender, 'ContextPreprocessor')
                    table.set_result(i + 1, 'gender_source', 'ContextPreprocessor', 'GenderPreprocessor')
                    if self.debug:
                        print(f"  ✅ Row {i + 1}: '{word}' → using person context gender: {current_person_gender} (inherent: {inherent_gender})")
                elif global_gender:
                    # RULE: Gender in table comes ONLY from milon
                    # Global gender context should NOT be applied to regular words
                    # Only personal pronouns get gender from gender parameter (handled separately)
                    if self.debug:
                        print(f"  ⏭️ Row {i + 1}: '{word}' → skipping global gender context (gender comes only from milon)")
                elif inherent_gender:
                    # Store inherent gender when no global or person context
                    table.set_result(i + 1, 'gender', inherent_gender, 'GenderPreprocessor')
                    table.set_result(i + 1, 'gender_source', 'Inherent_Gender', 'GenderPreprocessor')
                    if self.debug:
                        print(f"  ✅ Row {i + 1}: '{word}' → inherent gender: {inherent_gender}")

    def _clean_word_for_gender_lookup(self, word):
        """Clean word for gender lookup by removing prefixes and punctuation"""
        if not word:
            return ""

        # SPECIAL CASES: Don't clean pronouns and person indicators
        if word == 'את':
            return word  # Keep את as-is when it's standalone (direct object marker)

        # Don't clean complete pronouns that happen to start with prefixes
        if word in ['אתה', 'אני', 'אנחנו', 'אתם', 'אתן', 'אַתְּ', 'שאתה', 'ואתה', 'כשאתה', 'שאַתְּ', 'ואַתְּ', 'כשאַתְּ', 'לך', 'את/ה']:
            return word  # Keep pronouns as-is (added את/ה back for person detection)

        # Remove common prefixes (both with and without hyphens)
        # Note: Longer prefixes must come first to avoid partial matches
        prefixes_to_remove = [
            'מה-', 'את-', 'על-', 'עם-', 'ב-', 'ל-', 'מ-', 'כ-', 'ש-', 'ה-', 'ו-',  # With hyphens
            'מֵהָ', 'את', 'על', 'עם', 'ב', 'ל', 'מְ', 'כ', 'ש', 'ה', 'ו'              # Without hyphens
        ]
        cleaned = word

        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        # Remove trailing punctuation
        cleaned = cleaned.rstrip('.,;:!?()[]{}״"\'`')

        return cleaned

    def _calculate_word_position(self, text, word, word_index):
        """Calculate the position of a word in the original text"""
        if not text or not word:
            return 0
        try:
            # Split text into words and find cumulative position
            words = text.split()
            if word_index >= len(words):
                return len(text)

            # Find position by joining words up to current index
            text_before_word = ' '.join(words[:word_index])
            position = len(text_before_word)
            if position > 0:  # Add space if not at beginning
                position += 1
            return position
        except:
            return text.find(word) if word in text else 0

    def _lookup_gender_from_sources(self, clean_word, original_word, text=None, word_position=None):
        """Look up gender from milon and noun gender information"""

        # Simplified: milon-based approach handles את automatically

        # Try noun gender information first (most reliable)
        if hasattr(self, '_hebrew_nouns_gender') and clean_word in self._hebrew_nouns_gender:
            gender_info = self._hebrew_nouns_gender[clean_word]

            if gender_info == 'm':
                return 'm'
            elif gender_info == 'f':
                return 'f'

        # Try milon lookup (if word exists in dictionary)
        if hasattr(self, 'optimized_tables') and self.optimized_tables:
            # Search in milon data
            try:
                for table_name, table_data in self.optimized_tables.items():
                    if hasattr(table_data, 'data'):
                        # Look for the word in the milon
                        data_df = getattr(table_data, 'data')  # type: ignore
                        matches = data_df[data_df.iloc[:, 0] == clean_word]
                        if not matches.empty:
                            # Found word in milon - try to extract gender from the conversion
                            conversion = matches.iloc[0, 1] if len(matches.columns) > 1 else ""

                            # Only apply conversion-based heuristic if conversion is meaningful
                            if conversion and conversion.strip() and conversion.strip() not in ['', ' ', 'None']:
                                # Basic heuristic: if conversion contains feminine endings, it's likely feminine
                                if any(ending in conversion for ending in ['ת', 'ה', 'ות']):
                                    return 'f'
                                # Don't assume masculine - let it fall through to explicit gender columns
                            else:
                                # Check Zachar/Nekeva columns directly
                                zachar_col = matches.iloc[0].get('Zachar', '') if 'Zachar' in matches.columns else ''
                                nekeva_col = matches.iloc[0].get('Nekeva', '') if 'Nekeva' in matches.columns else ''


                                # Only one column has data → infer gender from that
                                if nekeva_col and str(nekeva_col).strip() and str(nekeva_col).strip() != 'nan':
                                    return 'f'
                                elif zachar_col and str(zachar_col).strip() and str(zachar_col).strip() != 'nan':
                                    return 'm'
                                # If no gender info in milon, fall through to heuristics below
            except Exception:
                pass  # Ignore milon lookup errors

        return None  # Gender unknown - only use explicit gender information from libraries



    def find_context_gender(self, table, current_row_index, search_radius=5):
        """Find gender context - search 4 lines from current line to find first gender"""
        if not table or not table.rows:
            return None

        # First try current row
        if current_row_index < len(table.rows):
            current_gender = table.rows[current_row_index].get('gender')
            if current_gender and current_gender not in ['None', '']:
                if self.debug:
                    print(f"    [CONTEXT] Using current row {current_row_index + 1} gender: '{current_gender}'")
                return current_gender

        # Search 4 lines from current line (current + next 3)
        search_end = min(current_row_index + 4, len(table.rows))
        for i in range(current_row_index, search_end):
            gender = table.rows[i].get('gender')
            if gender and gender not in ['None', '']:
                if self.debug:
                    print(f"    [CONTEXT] Found first gender in 4-line range: row {i + 1} has gender '{gender}'")
                return gender

        if self.debug:
            print(f"    [CONTEXT] No gender found in 4 lines from row {current_row_index + 1}")
        return None

    def _word_will_likely_change_in_milon(self, word):
        """Check if a word is likely to be converted/changed in milon processing"""
        if not word:
            return False

        # Simple heuristic: if word exists in milon and has actual conversion data, it will likely change
        # This is a conservative check to avoid unnecessary gender assignments
        try:
            if hasattr(self, 'optimized_tables') and self.optimized_tables:
                for table_name, table_data in self.optimized_tables.items():
                    if hasattr(table_data, 'data'):
                        data_df = getattr(table_data, 'data')  # type: ignore
                        matches = data_df[data_df.iloc[:, 0] == word]
                        if not matches.empty:
                            conversion = matches.iloc[0, 1] if len(matches.columns) > 1 else ""
                            # Word will likely change if it has actual conversion data (not empty or same as original)
                            if conversion and conversion.strip() and conversion.strip() != word:
                                return True
        except Exception:
            pass

        # Common prepositions and function words that typically don't change
        neutral_words = {'עד', 'דרך', 'אל', 'על', 'מ', 'ל', 'ב', 'כ', 'ש', 'ה'}
        return word not in neutral_words

    def _pure_milon_lookup(self, text, gender_context=None):
        """Pure dictionary lookup without any number processing or other transformations"""
        try:
            # Clean search word
            search_word = text.strip()
            if not search_word:
                return None



            # CRITICAL FIX: Only apply gender parameter conversion when person indicators are detected
            # Check if there are any person indicators in the global context
            has_person_indicator = (
                self.text_processor.global_context.get('detected_gender') is not None or
                self.text_processor.global_context.get('second_person_detected') or
                self.text_processor.global_context.get('force_maintain_gender')
            )

            if has_person_indicator:
                # Person indicators detected - apply gender parameter conversion
                source_gender = self.gender[0] if len(self.gender) >= 1 else 'f'
                target_gender = self.gender[2] if len(self.gender) >= 3 else source_gender
            else:
                # No person indicators - use same gender for both (no conversion)
                # Default to masculine form when no person indicators are detected
                default_gender = 'm'  # Use masculine as default when no person context
                source_gender = default_gender
                target_gender = default_gender
                if self.debug:
                    print(f"[DEBUG] _pure_milon_lookup: No person indicators detected, skipping gender conversion ({self.gender})")

            source_tts_gender = "female" if source_gender == "f" else "male"
            target_tts_gender = "female" if target_gender == "f" else "male"

            # Direct milon search using the same logic as search_in_milon but with table gender context
            marked_replacement, replacement, tts_gender, found, is_exclusive_gender = self.search_in_milon(
                search_word=search_word,
                person_context=None,
                source_gender=source_gender,
                target_gender=target_gender,
                source_tts_gender=source_tts_gender,
                target_tts_gender=target_tts_gender,
                gender_context=gender_context,  # Use the gender context from the table
                existing_highlight_blocks=[],
                text=search_word,
                leading_punct="",
                trailing_punct=""
            )

            if found:  # Return result if word was found in dictionary, regardless of conversion
                # Convert tts_gender to single letter format for table
                gender_letter = 'm' if tts_gender == "male" else 'f' if tts_gender == "female" else None
                return (search_word, replacement, gender_letter, is_exclusive_gender)

            # Dual gender forms should be handled through the milon dictionary
            # Removed hardcoded /ה handling - all slash forms should be in milon

            return None

        except Exception as e:
            if self.debug:
                print(f"  Pure milon lookup error for '{text}': {e}")
            return None

    def run_milon_processor(self, table):
        """Run dictionary processor on table using original algorithm approach"""
        if self.debug:
            self._debug_print("[MILON] Running MilonProcessor...")

        i = 0  # 0-based indexing for simplicity
        while i < len(table.rows):
            # Check if current word is already part of a pattern/conversion (has span markers or pattern results)
            current_row = table.rows[i]

            # Skip if word already has pattern result or is marked as span
            if (current_row.get('pattern') and
                (not current_row['pattern'].isdigit() or len(current_row['pattern']) > 1)):
                if self.debug:
                    print(f"  ⏭️  Skipping '{current_row['source']}' (row {i+1}) - already has pattern result")
                i += 1
                continue


            # Skip if word has span marker in heb2num or pattern column
            heb2num_val = current_row.get('heb2num', '')
            pattern_val = current_row.get('pattern', '')
            if ((heb2num_val and heb2num_val.isdigit() and len(heb2num_val) == 1) or
                (pattern_val and pattern_val.isdigit() and len(pattern_val) == 1)):
                if self.debug:
                    print(f"  ⏭️  Skipping '{current_row['source']}' (row {i+1}) - has span marker")
                i += 1
                continue

            # GREEDY LONGEST-MATCH with early exit optimization
            found_match = False
            best_match = None
            best_word_count = 0

            # Step 1: Check if first word exists (early exit check)
            if i < len(table.rows):
                first_word = table.rows[i]['source']

                # Slash forms (את/ה, מאמינ/ה, etc.) should be handled through milon dictionary
                # Removed hardcoded slash handling - all slash forms should be in milon
                                # Determine gender context with priority: 1) Explicit pronouns, 2) Local person context, 3) Global context
                preprocessed_gender = table.rows[i].get('gender')
                global_gender = self.text_processor.global_context.get('detected_gender')

                # Check if this word is an explicit pronoun that should not be overridden
                is_explicit_pronoun = (first_word in PERSON_INDICATORS['second']['male'] or
                                     first_word in PERSON_INDICATORS['second']['female'] or
                                     first_word in PERSON_INDICATORS['second']['neutral'] or
                                     first_word in PERSON_INDICATORS['first'])

                # Check if previous word was a first person pronoun (for local context)
                local_first_person_context = None
                if i > 0:
                    prev_word = table.rows[i-1]['source']
                    if prev_word in PERSON_INDICATORS['first']:
                        # Previous word is first person, so this word should use source gender
                        source_gender_letter = self.gender[0] if len(self.gender) >= 1 else 'f'
                        local_first_person_context = source_gender_letter
                        if self.debug:
                            print(f"[LOCAL_CONTEXT] Row {i+1} '{first_word}': following first person '{prev_word}', using source gender '{source_gender_letter}'")

                # Check if this word represents a second person context that should use target gender
                local_second_person_context = None
                if first_word == 'את' or first_word.endswith('את'):
                    # In conversion modes, את/second person should use target gender
                    target_gender_letter = self.gender[2] if len(self.gender) >= 3 else self.gender[0]
                    local_second_person_context = target_gender_letter
                    if self.debug:
                        print(f"[SECOND_PERSON_CONTEXT] Row {i+1} '{first_word}': second person should use target gender '{target_gender_letter}' in {self.gender} mode")

                # Check if we're within range of a recent second person pronoun
                recent_second_person_context = None
                for prev_i in range(max(0, i-3), i):  # Check up to 3 words back
                    prev_word = table.rows[prev_i]['source']
                    if prev_word == 'את' or prev_word.endswith('את') or prev_word in PERSON_INDICATORS['second']['male'] or prev_word in PERSON_INDICATORS['second']['female'] or prev_word in PERSON_INDICATORS['second']['neutral']:
                        # Found recent second person pronoun - use target gender
                        target_gender_letter = self.gender[2] if len(self.gender) >= 3 else self.gender[0]
                        recent_second_person_context = target_gender_letter
                        if self.debug:
                            print(f"[RECENT_SECOND_PERSON] Row {i+1} '{first_word}': using target gender '{target_gender_letter}' from recent second person '{prev_word}' at row {prev_i+1}")
                        break

                if is_explicit_pronoun:
                    gender_context = preprocessed_gender
                    if self.debug and global_gender:
                        print(f"[PRONOUN_PRESERVED] Row {i+1} '{first_word}': keeping explicit pronoun gender '{preprocessed_gender}', ignoring global context '{global_gender}'")
                elif local_first_person_context:
                    gender_context = local_first_person_context
                elif local_second_person_context:
                    gender_context = local_second_person_context
                elif recent_second_person_context:
                    gender_context = recent_second_person_context
                elif global_gender:
                    gender_context = 'm' if global_gender == 'male' else 'f'
                    if self.debug and preprocessed_gender and preprocessed_gender != gender_context:
                        print(f"[GENDER_OVERRIDE] Row {i+1} '{first_word}': using global context '{gender_context}' instead of preprocessed '{preprocessed_gender}'")
                else:
                    gender_context = preprocessed_gender
                first_word_result = self._pure_milon_lookup(first_word, gender_context=gender_context)

                # If first word doesn't exist (even after prefix removal), early exit
                if not first_word_result:
                    clean_word = self._clean_word_for_gender_lookup(first_word)
                    if clean_word != first_word:
                        first_word_result = self._pure_milon_lookup(clean_word, gender_context=gender_context)

                    if not first_word_result:
                        i += 1  # Early exit - move to next word
                        continue
                    else:
                        # Cleaned word exists, reconstruct with original prefix
                        removed_prefix = first_word[:len(first_word) - len(clean_word)]
                        replacement = first_word_result[1]  # Extract replacement from tuple
                        best_match = (first_word_result[0], removed_prefix + (replacement or ""), first_word_result[2], first_word_result[3])
                        best_word_count = 1
                else:
                    # First word exists, save as potential match
                    best_match = first_word_result
                    best_word_count = 1

            # Step 2-4: Try progressively longer combinations (1→2→3)
            for word_count in [1, 2, 3]:
                if i + word_count <= len(table.rows):
                    # Check if any of the words in this combination have pattern results
                    skip_combination = False
                    for j in range(word_count):
                        check_row = table.rows[i + j]
                        if (check_row.get('pattern') and
                            (not check_row['pattern'].isdigit() or len(check_row['pattern']) > 1)):
                            skip_combination = True
                            break

                    if skip_combination:
                        continue  # Try smaller word count

                    # Create combination
                    words = [table.rows[i + j]['source'] for j in range(word_count)]
                    combination = ' '.join(words)

                    # SMART SKIP: Quick dictionary check before full milon lookup
                    normalized_search = self._TextCleaningUtils.normalize_quotes(combination).lower()
                    quick_hit = hasattr(self, '_milon_dict') and normalized_search in self._milon_dict

                    # Check dictionary using PURE milon search (no number processing)
                    try:
                                                # Get gender context with priority: 1) Explicit pronouns, 2) Local person context, 3) Global context
                        preprocessed_gender = table.rows[i].get('gender') if i < len(table.rows) else None
                        global_gender = self.text_processor.global_context.get('detected_gender')

                        # Check if first word is an explicit pronoun that should not be overridden
                        first_word_in_combo = table.rows[i]['source'] if i < len(table.rows) else ""
                        is_explicit_pronoun = (first_word_in_combo in PERSON_INDICATORS['second']['male'] or
                                             first_word_in_combo in PERSON_INDICATORS['second']['female'] or
                                             first_word_in_combo in PERSON_INDICATORS['second']['neutral'] or
                                             first_word_in_combo in PERSON_INDICATORS['first'])

                                                # Check if previous word was a first person pronoun (for local context)
                        local_first_person_context = None
                        if i > 0:
                            prev_word = table.rows[i-1]['source']
                            if prev_word in PERSON_INDICATORS['first']:
                                # Previous word is first person, so this word should use source gender
                                source_gender_letter = self.gender[0] if len(self.gender) >= 1 else 'f'
                                local_first_person_context = source_gender_letter

                        # Check if this word represents a second person context that should use target gender
                        local_second_person_context = None
                        if first_word_in_combo == 'את' or first_word_in_combo.endswith('את'):
                            # In conversion modes, את/second person should use target gender
                            target_gender_letter = self.gender[2] if len(self.gender) >= 3 else self.gender[0]
                            local_second_person_context = target_gender_letter

                        # Check if we're within range of a recent second person pronoun
                        recent_second_person_context = None
                        for prev_i in range(max(0, i-3), i):  # Check up to 3 words back
                            prev_word = table.rows[prev_i]['source']
                            if prev_word == 'את' or prev_word.endswith('את') or prev_word in PERSON_INDICATORS['second']['male'] or prev_word in PERSON_INDICATORS['second']['female'] or prev_word in PERSON_INDICATORS['second']['neutral']:
                                # Found recent second person pronoun - use target gender
                                target_gender_letter = self.gender[2] if len(self.gender) >= 3 else self.gender[0]
                                recent_second_person_context = target_gender_letter
                                break

                        if is_explicit_pronoun:
                            gender_context = preprocessed_gender
                        elif local_first_person_context:
                            gender_context = local_first_person_context
                        elif local_second_person_context:
                            gender_context = local_second_person_context
                        elif recent_second_person_context:
                            gender_context = recent_second_person_context
                        elif global_gender:
                            gender_context = 'm' if global_gender == 'male' else 'f'
                        else:
                            gender_context = preprocessed_gender

                        dict_result = self._pure_milon_lookup(combination, gender_context=gender_context)

                        if dict_result:
                            # Found a match - save as best match (longer matches override shorter ones)
                            best_match = dict_result
                            best_word_count = word_count
                            if self.debug:
                                print(f"  [MATCH] Found {word_count}-word match: '{combination}'")
                        # Continue checking longer combinations (don't break)

                    except Exception as e:
                        if self.debug:
                            print(f"  Milon error for '{combination}': {e}")

            # Step 5: Process the best match found
            if best_match:
                if len(best_match) == 4:
                    original_word, replacement, gender_from_milon, is_exclusive_gender = best_match  # type: ignore
                elif len(best_match) == 3:
                    original_word, replacement, gender_from_milon = best_match  # type: ignore
                    is_exclusive_gender = False  # Default for backward compatibility
                else:
                    # Fallback for backwards compatibility
                    original_word, replacement = best_match  # type: ignore
                    gender_from_milon = None
                    is_exclusive_gender = False

                if self.debug:
                    print(f"  [BEST] Using {best_word_count}-word match: '{original_word}' → '{replacement}'")

                # Store the result in the first word's row
                if replacement and replacement != current_row['source']:
                    table.set_result(i + 1, 'milon', replacement, 'MilonProcessor')

                # IMMEDIATE GLOBAL CONTEXT UPDATE: Check if replacement contains pronouns
                if replacement and replacement != original_word:
                    # Check if replacement contains masculine pronouns
                    for male_pronoun in PERSON_INDICATORS['second']['male']:
                        if male_pronoun in replacement:
                            self.text_processor.global_context['detected_gender'] = 'male'
                            self.text_processor.global_context['second_person_detected'] = True
                            self.text_processor.global_context['detected_in_word'] = replacement
                            self.text_processor.global_context['force_maintain_gender'] = True
                            if self.debug:
                                print(f"[GLOBAL_UPDATE] Updated global context to 'male' from replacement: '{replacement}'")
                            break

                    # Check if replacement contains feminine pronouns (with nikud)
                    else:
                        for female_pronoun in PERSON_INDICATORS['second']['female']:
                            if female_pronoun in replacement:
                                self.text_processor.global_context['detected_gender'] = 'female'
                                self.text_processor.global_context['second_person_detected'] = True
                                self.text_processor.global_context['detected_in_word'] = replacement
                                self.text_processor.global_context['force_maintain_gender'] = True
                                if self.debug:
                                    print(f"[GLOBAL_UPDATE] Updated global context to 'female' from replacement: '{replacement}'")
                                break

                # RULE: Only assign gender to words that actually change during milon processing
                # If word stays the same (עד → עד), don't assign gender
                word_actually_changed = (replacement != original_word)

                # UPDATED RULE: Assign gender if word changes AND has gender from either:
                # 1) Explicit gender in main milon (זכר/נקבה columns) OR
                # 2) Gender information in Noun_Genders sheet
                noun_genders_gender = None
                if word_actually_changed and not (gender_from_milon and is_exclusive_gender):
                    # Check Noun_Genders sheet for gender information
                    clean_word_for_lookup = self._clean_word_for_gender_lookup(original_word)
                    if clean_word_for_lookup in self._hebrew_nouns_gender:
                        noun_genders_gender = self._hebrew_nouns_gender[clean_word_for_lookup]
                        if self.debug:
                            print(f"[DEBUG] Found gender in Noun_Genders for '{original_word}': {noun_genders_gender}")

                if word_actually_changed and gender_from_milon and is_exclusive_gender:
                    # Priority 1: Explicit gender in main milon
                    if self.debug:
                        print(f"[DEBUG] Updating Gender column for row {i + 1} with milon explicit gender: {gender_from_milon}")
                    table.set_result(i + 1, 'gender', gender_from_milon, 'MilonProcessor')
                    table.set_result(i + 1, 'gender_source', 'Milon', 'MilonProcessor')
                elif word_actually_changed and noun_genders_gender:
                    # Priority 2: Gender from Noun_Genders sheet
                    if self.debug:
                        print(f"[DEBUG] Updating Gender column for row {i + 1} with Noun_Genders gender: {noun_genders_gender}")
                    table.set_result(i + 1, 'gender', noun_genders_gender, 'MilonProcessor')
                    table.set_result(i + 1, 'gender_source', 'Noun_Genders', 'MilonProcessor')
                elif not word_actually_changed:
                    if self.debug:
                        print(f"[DEBUG] NOT updating Gender column for row {i + 1} - word unchanged ('{original_word}' → '{replacement}')")
                elif gender_from_milon and not is_exclusive_gender:
                    if self.debug:
                        print(f"[DEBUG] NOT updating Gender column for row {i + 1} - no explicit gender in milon (זכר/נקבה columns)")
                elif word_actually_changed and not gender_from_milon and not noun_genders_gender:
                    if self.debug:
                        print(f"[DEBUG] NOT updating Gender column for row {i + 1} - word changed but no gender info in milon or Noun_Genders")

                                # GENDER CONSISTENCY FIX: Set persistent gender context for subsequent words
                # This ensures subsequent words use consistent gender in f2m/m2f conversion modes

                # Case 1: Word was replaced in milon (both forms exist, one was chosen)
                should_set_context = False
                chosen_gender = None
                context_reason = ""

                if replacement != original_word and gender_from_milon and not is_exclusive_gender:
                    should_set_context = True
                    chosen_gender = gender_from_milon  # Use 'm' or 'f' directly
                    context_reason = f"milon replacement '{original_word}' → '{replacement}'"

                # Case 2: Masculine pronoun that establishes gender context (like אתה)
                elif original_word in PERSON_INDICATORS['second']['male']:
                    should_set_context = True
                    chosen_gender = 'm'
                    context_reason = f"masculine pronoun '{original_word}'"

                # Case 3: Feminine pronoun that establishes gender context (like אַתְּ)
                elif original_word in PERSON_INDICATORS['second']['female']:
                    should_set_context = True
                    chosen_gender = 'f'
                    context_reason = f"feminine pronoun '{original_word}'"

                # Case 4: Check if REPLACEMENT contains pronouns that should update global context
                elif replacement and replacement != original_word:
                    # Check if replacement contains masculine pronouns
                    for male_pronoun in PERSON_INDICATORS['second']['male']:
                        if male_pronoun in replacement:
                            should_set_context = True
                            chosen_gender = 'm'
                            context_reason = f"replacement contains masculine pronoun '{male_pronoun}': '{original_word}' → '{replacement}'"
                            # Also update global context
                            self.text_processor.global_context['detected_gender'] = 'male'
                            self.text_processor.global_context['second_person_detected'] = True
                            self.text_processor.global_context['detected_in_word'] = replacement
                            self.text_processor.global_context['force_maintain_gender'] = True
                            if self.debug:
                                print(f"[GLOBAL_UPDATE] Updated global context to 'male' from replacement: '{replacement}'")
                            break

                    # Check if replacement contains feminine pronouns (with nikud)
                    if not should_set_context:
                        for female_pronoun in PERSON_INDICATORS['second']['female']:
                            if female_pronoun in replacement:
                                should_set_context = True
                                chosen_gender = 'f'
                                context_reason = f"replacement contains feminine pronoun '{female_pronoun}': '{original_word}' → '{replacement}'"
                                # Also update global context
                                self.text_processor.global_context['detected_gender'] = 'female'
                                self.text_processor.global_context['second_person_detected'] = True
                                self.text_processor.global_context['detected_in_word'] = replacement
                                self.text_processor.global_context['force_maintain_gender'] = True
                                if self.debug:
                                    print(f"[GLOBAL_UPDATE] Updated global context to 'female' from replacement: '{replacement}'")
                                break

                if should_set_context and chosen_gender:
                    # Set gender context for next few words to maintain consistency
                    for next_row_idx in range(i + best_word_count + 1, min(i + best_word_count + 4, len(table.rows) + 1)):
                        if next_row_idx <= len(table.rows):
                            next_word = table.rows[next_row_idx - 1]['source'] if next_row_idx - 1 < len(table.rows) else ""
                            # Gender forms should be determined by milon dictionary, not slash patterns
                            existing_gender = table.rows[next_row_idx - 1].get('gender') if next_row_idx - 1 < len(table.rows) else None
                            # Skip standalone numbers - they should default to feminine, not inherit previous word's gender
                            is_standalone_number = next_word.isdigit()
                            # RULE: Only apply gender consistency to words that will actually be processed/changed
                            # Skip gender assignment for words that don't change (like prepositions עד, דרך)
                            will_be_processed = (
                                self._word_will_likely_change_in_milon(next_word)  # Words likely to be converted
                            )

                            if next_word and not existing_gender and not is_standalone_number and will_be_processed:
                                table.set_result(next_row_idx, 'gender', chosen_gender, 'GenderConsistency')
                                table.set_result(next_row_idx, 'gender_source', 'PreviousWordContext', 'GenderConsistency')
                                if self.debug:
                                    action = "Overriding inherent" if existing_gender else "Setting"
                                    print(f"[DEBUG] *** GENDER CONSISTENCY: {action} row {next_row_idx} ('{next_word}') gender to '{chosen_gender}' for consistency with {context_reason}")
                            elif next_word and not will_be_processed and self.debug:
                                print(f"[DEBUG] *** SKIPPING GENDER CONSISTENCY for row {next_row_idx} ('{next_word}') - word unlikely to change")

                # Mark remaining words as consumed with span markers
                for j in range(1, best_word_count):
                    table.set_result(i + j + 1, 'milon', str(best_word_count), 'MilonProcessor')

                # Move past all processed words
                i += best_word_count
                found_match = True
            else:
                # No match found, increment by 1
                i += 1

    def run_number_processor_original_text(self, table):
        """Run decimal currency AND regular number-noun processing on ORIGINAL TEXT before milon modifies currency words"""
        if self.debug:
            print("[NUMBER] Running EARLY NumberProcessor on ORIGINAL TEXT...")

        # Step 1: Get original text from table
        original_text = ' '.join([row['source'] for row in table.rows])

        if self.debug:
            print(f"  Original text for early processing: '{original_text}'")

        # Step 2: Run number processing from the old algorithm
        processed_text, number_changes = self.process_numbers_with_noun_context(original_text, track_changes=True)

        if self.debug:
            print(f"  Early processing result: '{processed_text}'")
            print(f"  Early number changes: {number_changes}")

        # Step 3: Store ALL number conversion results in the table
        # Note: Decimal currency results are already captured via self.current_processing_table
        # Now we need to handle regular number-noun conversions too
        self._store_early_number_changes_in_table(table, original_text, number_changes)

    def _store_early_number_changes_in_table(self, table, original_text, number_changes):
        """Store early number conversion results in the table"""
        if not number_changes:
            if self.debug:
                print("  No early number changes to store")
            return

        if self.debug:
            print(f"  Storing {len(number_changes)} early number changes in table...")

        for original_phrase, converted_phrase in number_changes.items():
            if self.debug:
                print(f"    Early processing: '{original_phrase}' → '{converted_phrase}'")

            # For single number conversions (like "2500" → "אַלְפַּיִם חֲמֵשׁ מֵאוֹת")
            if original_phrase.isdigit():
                # Find the number in the table
                for i, row in enumerate(table.rows):
                    if row['source'] == original_phrase:
                        table.set_result(i + 1, 'heb2num', converted_phrase, 'EarlyNumberProcessor')
                        if self.debug:
                            print(f"    ✅ Stored early number conversion: row {i + 1} = '{converted_phrase}'")
                        break

    def run_number_processor(self, table):
        """Run OLD number processor on FULL reconstructed text - store results in heb2num column"""
        if self.debug:
            print("[NUMBER] Running NumberProcessor on RECONSTRUCTED TEXT from table...")

        # Step 1: Reconstruct input text from table (rightmost non-null column)
        reconstructed_text = self._reconstruct_text_from_table_rightmost(table)
        original_text = ' '.join([row['source'] for row in table.rows])

        if self.debug:
            print(f"  Original text: '{original_text}'")
            print(f"  Reconstructed text: '{reconstructed_text}'")

        # Step 2: Run OLD heb2num algorithm on the reconstructed text
        processed_text, number_changes = self.process_numbers_with_noun_context(reconstructed_text, track_changes=True)

        if self.debug:
            print(f"  Number processing result: '{processed_text}'")
            print(f"  Number changes: {number_changes}")

        # Step 3: Store results back in table
        self._store_number_changes_in_table_v2(table, reconstructed_text, number_changes)

    def _reconstruct_text_from_table_rightmost(self, table):
        """Reconstruct text using rightmost non-null column value for each row, skipping span markers"""
        words = []
        for row in table.rows:
            # Check columns from right to left: heb2num, milon, source
            word_to_use = None
            for column in ['heb2num', 'milon', 'source']:
                value = row.get(column, '')
                if value:
                    # Skip span markers (single digits like "2", "3")
                    if value.isdigit() and len(value) == 1:
                        continue  # This is a span marker, skip this row completely
                    word_to_use = value
                    break

            # Only add words that are not span markers
            if word_to_use:
                words.append(word_to_use)

        return ' '.join(words)

    def _store_number_changes_in_table_v2(self, table, reconstructed_text, number_changes):
        """Store heb2num results back in table, mapping changes to original source rows"""
        if not number_changes:
            if self.debug:
                print("  No number changes to store")
            return

        # Build mapping from reconstructed text words back to original table rows
        reconstructed_words = reconstructed_text.split()
        word_to_table_mapping = {}

        # Track which table rows contributed to the reconstructed text
        reconstructed_idx = 0
        for table_idx, row in enumerate(table.rows):
            # Check if this row has a milon span marker (skip those rows)
            milon_val = row.get('milon', '')
            if milon_val and milon_val.isdigit() and len(milon_val) == 1:
                continue  # Skip span marker rows

            # Map this reconstructed word back to table row
            if reconstructed_idx < len(reconstructed_words):
                word_to_table_mapping[reconstructed_idx] = table_idx
                reconstructed_idx += 1

        if self.debug:
            print("  📍 Word mapping: reconstructed → table")
            for rec_idx, table_idx in word_to_table_mapping.items():
                rec_word = reconstructed_words[rec_idx] if rec_idx < len(reconstructed_words) else "?"
                table_word = table.rows[table_idx]['source'] if table_idx < len(table.rows) else "?"
                print(f"    {rec_idx}: '{rec_word}' → row {table_idx}: '{table_word}'")

        # Process each number change
        for original_phrase, converted_phrase in number_changes.items():
            if self.debug:
                print(f"  Processing change: '{original_phrase}' → '{converted_phrase}'")

            # Find where this phrase starts in the reconstructed text
            original_words = original_phrase.split()
            converted_words = converted_phrase.split()

            phrase_start_idx = self._find_phrase_in_words(reconstructed_words, original_words)

            # If not found, try finding with currency normalization
            if phrase_start_idx == -1:
                phrase_start_idx = self._find_phrase_with_currency_normalization(reconstructed_words, original_words)

            if phrase_start_idx == -1:
                if self.debug:
                    print(f"    ⚠️  Could not find phrase '{original_phrase}' in reconstructed text")
                continue

            # Map to table row
            if phrase_start_idx in word_to_table_mapping:
                table_start_row = word_to_table_mapping[phrase_start_idx]

                if self.debug:
                    print(f"    📍 Found phrase at reconstructed pos {phrase_start_idx} → table row {table_start_row}")

                # Store the result in the table (heb2num column)
                print(f"[DEBUG_STORE] About to store: converted_words={len(converted_words)}, converted_phrase='{converted_phrase}', table_start_row={table_start_row}")

                if len(converted_words) == 1:
                    # Single word replacement
                    print(f"[DEBUG_STORE] Calling set_result for single word: row {table_start_row + 1}, column 'heb2num', value '{converted_phrase}'")
                    table.set_result(table_start_row + 1, 'heb2num', converted_phrase, 'NumberProcessor')
                    print(f"[DEBUG_CONVERSION] Original: '{original_phrase}' | Hebrew: '{converted_phrase}'")
                    if self.debug:
                        print(f"    ✅ Stored in table: row {table_start_row + 1} = '{converted_phrase}', span = 1")
                else:
                    # Multi-word replacement - store main result and span markers
                    print(f"[DEBUG_STORE] Calling set_result for multi-word: row {table_start_row + 1}, column 'heb2num', value '{converted_phrase}'")
                    table.set_result(table_start_row + 1, 'heb2num', converted_phrase, 'NumberProcessor')
                    print(f"[DEBUG_CONVERSION] Original: '{original_phrase}' | Hebrew: '{converted_phrase}'")
                    span_size = len(original_words)
                    for i in range(1, span_size):
                        next_table_row = table_start_row + i
                        if next_table_row < len(table.rows):
                            print(f"[DEBUG_STORE] Calling set_result for span marker: row {next_table_row + 1}, column 'heb2num', value '{str(span_size)}'")
                            table.set_result(next_table_row + 1, 'heb2num', str(span_size), 'NumberProcessor')
                    if self.debug:
                        print(f"    ✅ Stored in table: row {table_start_row + 1} = '{converted_phrase}', span = {span_size}")
            else:
                if self.debug:
                    print(f"    ⚠️  Could not map phrase position {phrase_start_idx} to table row")

        if self.debug:
            changes_applied = len([change for change in number_changes.items()
                                 if self._find_phrase_in_words(reconstructed_words, change[0].split()) != -1])
            print(f"  NumberProcessor completed. Applied {changes_applied} changes.")

    def _find_phrase_in_words(self, word_list, target_phrase):
        """Find the starting index of target_phrase in word_list. Returns -1 if not found."""
        if not target_phrase:
            return -1

        target_len = len(target_phrase)
        for i in range(len(word_list) - target_len + 1):
            if word_list[i:i + target_len] == target_phrase:
                return i
        return -1

    def _find_phrase_with_currency_normalization(self, word_list, target_phrase):
        """Find phrase allowing currency word equivalents (ש\"ח ↔ שקלים variants)"""
        if not target_phrase:
            return -1

        # Currency equivalents no longer needed - everything is standardized to שקלים early in pipeline
        currency_equivalents = {}

        target_len = len(target_phrase)
        for i in range(len(word_list) - target_len + 1):
            matches = True
            for j in range(target_len):
                word_in_list = word_list[i + j]
                target_word = target_phrase[j]

                # Check exact match first
                if word_in_list == target_word:
                    continue

                # Check currency equivalents
                equivalents = currency_equivalents.get(target_word, [])
                if word_in_list in equivalents:
                    continue

                # No match found
                matches = False
                break

            if matches:
                return i

        return -1

    def run_individual_number_processor(self, table):
        """Process individual numbers that haven't been converted yet"""
        import re
        if self.debug:
            print("[NUMBER] Running IndividualNumberProcessor...")

        for i, row in enumerate(table.rows, 1):
            word = row['source']

            # Skip if already has a result or is a span marker
            heb2num_value = row['heb2num']
            milon_value = row['milon']

            # Skip if has milon conversion (milon takes priority over automatic number conversion)
            if milon_value and milon_value not in ['None', '', None]:
                if self.debug:
                    print(f"  ⏭️  Skipping '{word}' (row {i}) - already has milon conversion: '{milon_value}'")
                continue

            # Skip if already has heb2num conversion (but allow override if gender context is available)
            if heb2num_value and heb2num_value != 'None':
                # Check if it's a span marker (single digit string like "2", "3")
                if heb2num_value.isdigit() and len(heb2num_value) == 1:
                    if self.debug:
                        print(f"  ⏭️  Skipping '{word}' (row {i}) - has span marker '{heb2num_value}'")
                    continue
                else:
                    # Don't override if it's already a complete percentage conversion
                    if 'אחוזים' in heb2num_value or 'אחוז' in heb2num_value:
                        if self.debug:
                            print(f"  ⏭️  Skipping '{word}' (row {i}) - already has percentage conversion: '{heb2num_value}'")
                        continue

                    # Don't override if it's already a date ordinal conversion (contains month names)
                    hebrew_months = ['ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני',
                                   'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר']
                    if any(month in heb2num_value for month in hebrew_months):
                        if self.debug:
                            print(f"  ⏭️  Skipping '{word}' (row {i}) - already has date ordinal conversion: '{heb2num_value}'")
                        continue

                    # Don't override if it contains ordinal patterns (ends with ordinal suffixes)
                    ordinal_patterns = ['ראשון', 'שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שביעי', 'שמיני', 'תשיעי', 'עשירי']
                    if any(ordinal in heb2num_value for ordinal in ordinal_patterns):
                        if self.debug:
                            print(f"  ⏭️  Skipping '{word}' (row {i}) - already has ordinal conversion: '{heb2num_value}'")
                        continue

                    # Check if next word has gender that could override the early processing
                    should_override = False
                    if word.isdigit() and i < len(table.rows):  # Check if there's a next row
                        next_row = table.rows[i]
                        # If next row is just a span marker (heb2num is single digit), keep existing conversion
                        next_row_span = next_row.get('heb2num')
                        if next_row_span and next_row_span.isdigit() and len(next_row_span) == 1:
                            should_override = False
                        else:
                            next_row_gender = next_row.get('gender')
                            if next_row_gender and next_row_gender not in ['None', '']:
                                should_override = True
                                if self.debug:
                                    print(f"  [OVERRIDE] Reprocessing '{word}' (row {i}) - next word has gender '{next_row_gender}' that may override early processing")

                    if not should_override:
                        # Already has a real Hebrew conversion, keep it
                        continue

            # Check if it's a number or prefixed number (like ב-3, ל-5)
            if word.isdigit():
                try:
                    number = int(word)

                    # Special handling for 7+ digit numbers: convert digit-by-digit as feminine
                    if len(word) >= 7:
                        hebrew_digits = self._convert_digits_number(word, 'נקבה')  # Always feminine for digit-by-digit
                        table.set_result(i, 'heb2num', hebrew_digits, 'IndividualNumberProcessor')
                        if self.debug:
                            print(f"  [DIGITS] 7+ digit number: '{word}' → '{hebrew_digits}' (row {i}, digit-by-digit feminine)")
                        continue  # Skip regular number processing

                    # Check if this is a year-like number (2000-2030) - force feminine and skip other gender detection
                    if 2000 <= number <= 2030:
                        gender_char = 'f'
                        gender_source = "year range (2000-2030) - forced feminine"
                        if self.debug:
                            print(f"  [YEAR_RANGE] Forcing feminine gender for year-like number '{number}' (row {i})")
                    else:
                        # NEW RULE: Use current row gender first, then next word gender, default to feminine
                        gender_char = 'f'  # Default to feminine
                        gender_source = "default feminine"

                        # Priority 1: Use current row gender
                        current_row_index = i - 1  # Convert to 0-based index
                        if current_row_index < len(table.rows):
                            current_row_gender = table.rows[current_row_index].get('gender')
                            if current_row_gender and current_row_gender not in ['None', '']:
                                gender_char = current_row_gender
                                gender_source = f"current row {i}"
                                if self.debug:
                                    print(f"  [CURRENT_ROW] Using gender from current row: '{gender_char}' for number '{word}' (row {i})")
                            else:
                                # Priority 2: Check next word gender (for patterns like "4 ביולי")
                                if i < len(table.rows):  # Check if there's a next row
                                    next_row_gender = table.rows[i].get('gender')  # i is already 1-based, so table.rows[i] is next row
                                    if next_row_gender and next_row_gender not in ['None', '']:
                                        gender_char = next_row_gender
                                        gender_source = f"next word (row {i + 1})"
                                        if self.debug:
                                            print(f"  [NEXT_WORD] Using gender from next word: '{gender_char}' for number '{word}' (row {i})")
                                    elif self.debug:
                                        print(f"  [DEFAULT] Current and next rows have no gender, using default feminine for number '{word}' (row {i})")
                                elif self.debug:
                                    print(f"  [DEFAULT] No next row, using default feminine for number '{word}' (row {i})")
                        elif self.debug:
                            print(f"  [DEFAULT] Row index out of range, using default feminine for number '{word}' (row {i})")

                        # Priority 3: Default feminine (already set)
                        if gender_char == 'f' and gender_source == "default feminine":
                            if self.debug:
                                print(f"  [PRIORITY 3] Using default feminine gender for number '{word}' (row {i}) - no context found")

                    # Check if next word ends with ם or ת (like אגורות format)
                    if i < len(table.rows):  # Check if there's a next word
                        next_word = table.rows[i]['source'] if i < len(table.rows) else ''  # i is 1-based, so table.rows[i] is the next word
                        if next_word.endswith('ם') or next_word.endswith('ת'):
                            # Use the gender of the following word (like שנים/אגורות) instead of context gender
                            next_word_gender = table.rows[i].get('gender') if i < len(table.rows) else None
                            if next_word_gender:
                                # The gender in the table is already 'm' or 'f'
                                gender_char = next_word_gender
                                if self.debug:
                                    print(f"  📝 Using regular format for '{word}' with gender from next word '{next_word}' (gender: {next_word_gender}) - like אגורות")
                            elif self.debug:
                                print(f"  📝 Using regular format for '{word}' because next word '{next_word}' ends with ם/ת (keeping gender: {gender_char}) - like אגורות")

                    # Use cardinal numbers for regular counting (like 5 צ׳קים)
                    # Only use ordinals for dates or specific sequence contexts
                    hebrew_number = self.heb2num(number, gender_char)
                    if self.debug:
                        print(f"  [CARDINAL] Using cardinal for number {number} with gender '{gender_char}' → '{hebrew_number}'")
                    table.set_result(i, 'heb2num', hebrew_number, 'IndividualNumberProcessor')
                    # Store the determined gender in the table for transparency
                    table.set_result(i, 'gender', gender_char, 'IndividualNumberProcessor')
                    table.set_result(i, 'gender_source', gender_source, 'IndividualNumberProcessor')
                    if self.debug:
                        print(f"  [NUMBER] Individual number: '{word}' → '{hebrew_number}' (row {i}, gender: {gender_char})")
                except Exception as e:
                    if self.debug:
                        print(f"  ❌ Error converting individual number '{word}': {e}")

            # Check if it's a prefixed number (like ב-3, ל-5, מ-7, ול-1000)
            elif re.match(r'^[א-ת]+-\d+$', word):
                try:
                    prefix = word.split('-')[0] + '-'
                    number_str = word.split('-')[1]
                    number = int(number_str)

                    # NEW RULE: Use 4-line gender search for prefixed numbers
                    current_row_index = i - 1  # Convert to 0-based index
                    gender_char = self.find_context_gender(table, current_row_index) or 'f'  # Default to feminine if no gender found

                    if self.debug:
                        print(f"  [4-LINE_SEARCH] Using 4-line gender search for prefixed number '{word}' (row {i}): gender = '{gender_char}'")

                    # Check if next word ends with ם or ת (like אגורות format)
                    if i < len(table.rows):  # Check if there's a next word
                        next_word = table.rows[i]['source'] if i < len(table.rows) else ''  # i is 1-based, so table.rows[i] is the next word
                        if next_word.endswith('ם') or next_word.endswith('ת'):
                            # Use the gender of the following word (like שנים/אגורות) instead of context gender
                            next_word_gender = table.rows[i].get('gender') if i < len(table.rows) else None
                            if next_word_gender:
                                # The gender in the table is already 'm' or 'f'
                                gender_char = next_word_gender
                                if self.debug:
                                    print(f"  📝 Using regular format for '{word}' with gender from next word '{next_word}' (gender: {next_word_gender}) - like אגורות")
                            elif self.debug:
                                print(f"  📝 Using regular format for '{word}' because next word '{next_word}' ends with ם/ת (keeping gender: {gender_char}) - like אגורות")

                    # Apply construct state rule for prefixed numbers
                    # For prefixed numbers, use construct state ONLY if the following noun starts with ה
                    # Example: ל-5 התשלומים (construct) vs ל-3 תשלומים (absolute)
                    next_word = table.rows[i]['source'] if i < len(table.rows) else ''
                    use_construct_state = next_word.startswith('ה')  # Check if next word starts with ה

                    hebrew_number = self.heb2num(number, gender_char, construct_state=use_construct_state)
                    clean_prefix = prefix.rstrip('-')  # Remove hyphen: ב- → ב
                    prefixed_hebrew = f"{clean_prefix}{hebrew_number}"

                    if self.debug:
                        print(f"  [DEFINITE_ARTICLE] Next word: '{next_word}', starts with ה: {next_word.startswith('ה')}, construct_state: {use_construct_state}")

                    table.set_result(i, 'heb2num', prefixed_hebrew, 'IndividualNumberProcessor')
                    if self.debug:
                        print(f"  [PREFIX] Prefixed number: '{word}' → '{prefixed_hebrew}' (row {i}, gender: {gender_char})")
                except Exception as e:
                    if self.debug:
                        print(f"  ❌ Error converting prefixed number '{word}': {e}")

    def run_pattern_processor(self, table):
        """Run pattern processor on table (dates, currency, etc.)"""
        if self.debug:
            print("[PATTERN] Running PatternProcessor...")

        # Process date patterns in the table
        self._process_date_patterns_in_table(table)

        # Process percentage patterns in the table
        self._process_percentage_patterns_in_table(table)

        # Future: Add currency, prefixed numbers, etc.

    def _process_date_patterns_in_table(self, table):
        """Process date patterns for unconsumed table rows"""
        import re

        # Define the key Hebrew prefix pattern for date processing
        hebrew_date_pattern = r'([א-ת]+)-(\d{1,2})/(\d{1,2})'

        for i, row in enumerate(table.rows, 1):
            if table.is_consumed(i):
                continue

            # Skip rows that already have a conversion in heb2num column
            if row.get('heb2num') and row['heb2num'] != 'None':
                if self.debug:
                    print(f"  ⏭️  Skipping '{row['source']}' (row {i}) - already has heb2num conversion")
                continue

            source_text = row['source']
            match = re.match(hebrew_date_pattern, source_text)

            if match:
                prefix = match.group(1)      # ה
                day_str = match.group(2)     # 10
                month_str = match.group(3)   # 04

                try:
                    day_num = int(day_str)
                    month_num = int(month_str)

                    if 1 <= day_num <= 31 and 1 <= month_num <= 12:
                        # Convert day to Hebrew using existing method
                        day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                        # Convert month number to Hebrew month name
                        month_name = self._hebrew_months.get(month_num, str(month_num))

                        # Format result with prefix
                        converted = f"{prefix}{day_text} ב{month_name}"

                        # Store result in table
                        table.set_result(i, 'pattern', converted, 'PatternProcessor_HebrewDate')
                        table.mark_consumed(i, i, 'HEBREW_DATE_PATTERN', 'PatternProcessor')

                        if self.debug:
                            print(f"[DEBUG] HEBREW DATE PATTERN REPLACEMENT: '{source_text}' -> '{converted}'")

                except (ValueError, KeyError):
                    continue

    def _process_percentage_patterns_in_table(self, table):
        """Process percentage patterns for unconsumed table rows"""
        import re

        # Process standalone percentage patterns: 12.5%
        standalone_pattern = r'^(\d+(?:\.\d+)?)%$'

        for i, row in enumerate(table.rows, 1):
            if table.is_consumed(i):
                continue
            source_text = row['source']
            match = re.match(standalone_pattern, source_text)
            if match:
                number_str = match.group(1)
                hebrew_percentage = self._convert_percentage_to_hebrew(number_str)
                converted = f"{hebrew_percentage}"
                table.set_result(i, 'pattern', converted, 'PatternProcessor_Percentage')
                table.mark_consumed(i, i, 'PERCENTAGE_STANDALONE', 'PatternProcessor')
                if self.debug:
                    print(f"[DEBUG] STANDALONE PERCENTAGE: '{source_text}' -> '{converted}'")
                if table:
                    self._store_conversion_in_table(table, source_text, converted, "PERCENTAGE_STANDALONE")

        # Process single-word patterns: ל-12.5%
        single_word_pattern = r'([א-ת]+)-(\d+(?:\.\d+)?)%'

        for i, row in enumerate(table.rows, 1):
            if table.is_consumed(i):
                continue

            source_text = row['source']
            match = re.match(single_word_pattern, source_text)

            if match:
                prefix = match.group(1)      # ל
                number_str = match.group(2)  # 12.5

                # Use existing method
                hebrew_percentage = self._convert_percentage_to_hebrew(number_str)
                attachable_prefixes = {'ל', 'ב', 'כ', 'מ', 'ו', 'ש'}
                spacer = '' if prefix in attachable_prefixes else ' '
                converted = f"{prefix}{spacer}{hebrew_percentage}"

                # Store result in table
                table.set_result(i, 'pattern', converted, 'PatternProcessor_Percentage')
                table.mark_consumed(i, i, 'PERCENTAGE_PATTERN', 'PatternProcessor')

                if self.debug:
                    print(f"[DEBUG] PERCENTAGE PATTERN REPLACEMENT: '{source_text}' -> '{converted}'")

                # Store directly in table if available
                if table:
                    self._store_conversion_in_table(table, source_text, converted, "PERCENTAGE_PATTERN")

        # Process two-word patterns: ל 9.5%
        i = 1
        while i <= len(table.rows) - 1:  # Need at least 2 rows
            # Skip if either word is already consumed
            if table.is_consumed(i) or table.is_consumed(i + 1):
                i += 1
                continue

            first_word = table.rows[i - 1]['source']    # Adjust for 0-based indexing
            second_word = table.rows[i]['source']       # Adjust for 0-based indexing

            # Check if it matches pattern: Hebrew word + percentage
            two_word_pattern = r'^([א-ת]+)$'
            percentage_pattern = r'^(\d+(?:\.\d+)?)%$'

            percentage_match = re.match(percentage_pattern, second_word)
            if re.match(two_word_pattern, first_word) and percentage_match:
                # Use milon result if available, otherwise use original word
                milon_result = table.rows[i - 1].get('milon')
                prefix = milon_result if milon_result and milon_result not in ['None', '', None] else first_word
                number_str = percentage_match.group(1)  # 9.5

                # Use existing method
                hebrew_percentage = self._convert_percentage_to_hebrew(number_str)
                attachable_prefixes = {'ל', 'ב', 'כ', 'מ', 'ו', 'ש'}
                spacer = '' if prefix in attachable_prefixes else ' '
                converted = f"{prefix}{spacer}{hebrew_percentage}"

                # Store result in first row
                table.set_result(i, 'pattern', converted, 'PatternProcessor_Percentage')

                # Clear any prior replacements on the numeric row to avoid duplicates
                if 0 <= (i) < len(table.rows):
                    for col in ['pattern', 'heb2num', 'milon']:
                        table.rows[i][col] = None
                table.mark_consumed(i, i + 1, 'PERCENTAGE_2WORD', 'PatternProcessor')

                if self.debug:
                    print(f"[DEBUG] 2-WORD PERCENTAGE PATTERN: '{first_word} {second_word}' -> '{converted}'")

                # Store directly in table if available
                if table:
                    self._store_conversion_in_table(table, f"{first_word} {second_word}", converted, "PERCENTAGE_2WORD")

                # Move past both words
                i += 2
            else:
                # No match, move to next position
                i += 1

    def _update_milon_gender_tracking(self, row, search_word):
        """Handle last_milon_gender tracking logic with new gender rules"""
        has_zachar = 'Zachar' in row.columns and pd.notna(row.iloc[0]['Zachar']) and str(row.iloc[0]['Zachar']).strip()
        has_nekeva = 'Nekeva' in row.columns and pd.notna(row.iloc[0]['Nekeva']) and str(row.iloc[0]['Nekeva']).strip()

        # NEW GENDER RULES:
        # Rule 1: If זכר ≠ null AND נקבה = null → gender = f
        if has_zachar and not has_nekeva:
            self.text_processor.last_milon_gender = "female"  # Changed from "male" to "female"
            if self.debug:
                print(f"[DEBUG] *** last_milon_gender SET to 'female' for '{search_word}' (Rule 1: has Zachar, no Nekeva)")
                self._debug(f"Setting last_milon_gender = 'female' based on {search_word} (Rule 1: has Zachar, no Nekeva)")
            return True  # Exclusive gender form found

        # Rule 2: If זכר = null AND נקבה ≠ null → gender = m
        elif has_nekeva and not has_zachar:
            self.text_processor.last_milon_gender = "male"  # Changed from "female" to "male"
            if self.debug:
                print(f"[DEBUG] *** last_milon_gender SET to 'male' for '{search_word}' (Rule 2: has Nekeva, no Zachar)")
                self._debug(f"Setting last_milon_gender = 'male' based on {search_word} (Rule 2: has Nekeva, no Zachar)")
            return True  # Exclusive gender form found

        # Rule 3: If both זכר AND נקבה = null
        elif not has_zachar and not has_nekeva:
            # Check column C (גוף) for gender
            has_guf = len(row.columns) > 2 and pd.notna(row.iloc[0].iloc[2])  # Column C (3rd column)
            if has_guf:
                guf_value = str(row.iloc[0].iloc[2]).strip()
                if guf_value == "זכר":
                    self.text_processor.last_milon_gender = "male"
                    if self.debug:
                        print(f"[DEBUG] *** last_milon_gender SET to 'male' for '{search_word}' (Rule 3: גוף = זכר)")
                        self._debug(f"Setting last_milon_gender = 'male' based on {search_word} (Rule 3: גוף = זכר)")
                    return True
                elif guf_value == "נקבה":
                    self.text_processor.last_milon_gender = "female"
                    if self.debug:
                        print(f"[DEBUG] *** last_milon_gender SET to 'female' for '{search_word}' (Rule 3: גוף = נקבה)")
                        self._debug(f"Setting last_milon_gender = 'female' based on {search_word} (Rule 3: גוף = נקבה)")
                    return True
                else:
                    if self.debug:
                        print(f"[DEBUG] *** last_milon_gender NOT SET for '{search_word}' (Rule 3: גוף = '{guf_value}', not זכר/נקבה)")
                    return False  # Leave gender empty
            else:
                if self.debug:
                    print(f"[DEBUG] *** last_milon_gender NOT SET for '{search_word}' (Rule 3: no גוף column or empty)")
                return False  # Leave gender empty

        # Both columns have values - no exclusive gender
        else:
            if self.debug:
                print(f"[DEBUG] *** last_milon_gender NOT SET for '{search_word}' - has both gender forms")
            return False  # No exclusive gender form

    def _search_sorted_table(self, search_table, normalized_search):
        """
        ULTRA-FAST dictionary lookup first, then fallback to pandas search
        Dictionary lookup is 288x faster than binary search!
        """
        target = normalized_search.lower()

        # Track search statistics for debug mode
        if hasattr(self, '_search_stats'):
            self._search_stats['total_searches'] += 1

        # ULTRA-FAST: Try dictionary lookup first (O(1) complexity)
        if hasattr(self, '_milon_dict') and target in self._milon_dict:
            if hasattr(self, '_search_stats'):
                self._search_stats['dict_hits'] += 1
            row_dict = self._milon_dict[target]
            # Convert dict back to DataFrame for compatibility
            return pd.DataFrame([row_dict])

        # Track dictionary miss
        if hasattr(self, '_search_stats'):
            self._search_stats['dict_misses'] += 1

        # Fallback to pandas search for edge cases
        if search_table.empty:
            return pd.DataFrame()

        # Track pandas search usage
        if hasattr(self, '_search_stats'):
            self._search_stats['pandas_searches'] += 1

        # Use linear search for small tables (faster due to less overhead)
        if len(search_table) < 50:
            return self._linear_search(search_table, target)
        else:
            return self._binary_search(search_table, target)

    def _linear_search(self, search_table, target):
        """Optimized linear search for small tables"""
        # Use vectorized operations instead of iterrows() for better performance
        if 'Original' not in search_table.columns:
            return pd.DataFrame()

        # Create normalized column if not exists
        if '_normalized' not in search_table.columns:
            search_table = search_table.copy()
            search_table['_normalized'] = search_table['Original'].astype(str).str.strip().apply(
                lambda x: self._TextCleaningUtils.normalize_quotes(x).lower() if x else ''
            )

        # Use vectorized comparison
        matches = search_table[search_table['_normalized'] == target]
        return matches.head(1)  # Return first match

    def _binary_search(self, search_table, target):
        """Binary search for large tables using pre-computed _sort_key column"""
        left, right = 0, len(search_table) - 1

        while left <= right:
            mid = (left + right) // 2
            mid_row = search_table.iloc[mid]

            # Use pre-computed _sort_key if available, otherwise compute it
            if '_sort_key' in mid_row:
                sort_key = mid_row['_sort_key']
            else:
                original_word = mid_row.get('Original', '')
                if not original_word:
                    left = mid + 1
                    continue
                sort_key = self._TextCleaningUtils.normalize_quotes(str(original_word).strip()).lower()

            if sort_key == target:
                return search_table[search_table.index == mid_row.name]
            elif sort_key < target:
                left = mid + 1
            else:
                right = mid - 1

        return pd.DataFrame()

    # === HELPER METHODS (Moved from removed classes) ===

    def _wrap_result(self, result):
        """Wrap result with highlighting markers"""
        return result

    def _convert_percentage_to_hebrew(self, number_str):
        """Convert percentage number to Hebrew according to Hebrew rules"""
        if '.' in number_str:
            whole_part, decimal_part = number_str.split('.')
            whole_num = int(whole_part)
            decimal_str = decimal_part
        else:
            whole_num = int(number_str)
            decimal_str = None

        if whole_num == 1 and not decimal_str:
            return "אחוז אחד"
        else:
            # Use absolute state for percentages since "אחוזים" has no definite article (ה)
            # Exception: number 2 always uses "שני" even in absolute state
            if whole_num == 2:
                whole_hebrew = "שני"  # Special case for 2
            else:
                whole_hebrew = self.heb2num(whole_num, 'm', construct_state=False)

            if not decimal_str or decimal_str == "0":
                return f"{whole_hebrew} אחוזים"
            else:
                if decimal_str == "25":
                    return f"{whole_hebrew} אחוזים ורבע"
                elif decimal_str == "5":
                    return f"{whole_hebrew} אחוזים וחצי"
                elif decimal_str == "75":
                    return f"{whole_hebrew} אחוזים ושלושה רבעים"
                else:
                    decimal_num = int(decimal_str)
                    # Decimal part uses absolute state
                    decimal_hebrew = self.heb2num(decimal_num, 'm', construct_state=False)
                    return f"{whole_hebrew} אחוזים נקודה {decimal_hebrew}"

    def _handle_percentage_pattern(self, match, number_str, hebrew_word=None, track_changes=False, number_changes=None):
        """Unified percentage pattern handler"""
        hebrew_result = self._convert_percentage_to_hebrew(number_str)

        if hebrew_word:
            hebrew_result = f"{hebrew_word} {hebrew_result}"

        if track_changes and number_changes is not None:
            number_changes[match.group(0)] = hebrew_result

        return self._wrap_result(hebrew_result)

    def _handle_number_noun_pattern(self, match, prefix, num, noun, track_changes=False, number_changes=None):
        """Unified number-noun pattern handler"""
        clean_noun = self._TextCleaningUtils.clean_noun_for_lookup(noun)

        current_gender = self._hebrew_nouns_gender.get(clean_noun, 'f')
        clean_prefix = self._TextCleaningUtils.clean_hebrew_prefix(prefix)

        if '.' in num:
            # Handle decimal numbers
            whole_part, decimal_part = num.split('.')
            # Check if this is Israeli currency using Excel data
            is_israeli_currency = self._hebrew_nouns_types.get(noun, '') == 'שקלים'

            if is_israeli_currency:
                whole_text = self.heb2num(int(whole_part), current_gender, construct_state=True)
                decimal_normalized = decimal_part.ljust(2, '0') if len(decimal_part) == 1 else decimal_part
                decimal_value = int(decimal_normalized)

                # Special case for 1 shekel: use singular "שקל אחד" instead of "אחד שקלים"
                if whole_part == "1":
                    if decimal_value == 0:
                        hebrew_num = f"{clean_prefix}שקל אחד"
                    elif decimal_value == 1:
                        # 1.01 שקלים -> שקל אחד ואגורה אחת
                        hebrew_num = f"{clean_prefix}שקל אחד ואגורה אחת"
                    elif decimal_value == 2:
                        # 1.02 שקלים -> שקל אחד ושתי אגורות
                        hebrew_num = f"{clean_prefix}שקל אחד ושתי אגורות"
                    else:
                        decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                        hebrew_num = f"{clean_prefix}שקל אחד ו{decimal_text} אגורות"
                else:
                    if decimal_value == 0:
                        hebrew_num = f"{clean_prefix}{whole_text} שקלים"
                    elif decimal_value == 1:
                        # Special case for 1 agora: "אגורה אחת" instead of "אחת אגורות"
                        hebrew_num = f"{clean_prefix}{whole_text} שקלים ואגורה אחת"
                    elif decimal_value == 2:
                        # Special case for 2 agorot: "שתי אגורות" instead of "שתיים אגורות"
                        hebrew_num = f"{clean_prefix}{whole_text} שקלים ושתי אגורות"
                    else:
                        decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                        hebrew_num = f"{clean_prefix}{whole_text} שקלים ו{decimal_text} אגורות"
            else:
                whole_text = self.heb2num(int(whole_part), current_gender)
                decimal_text = self.heb2num(int(decimal_part), 'f')
                hebrew_num = f"{clean_prefix}{whole_text} {noun} נקודה {decimal_text}"
        else:
            # Handle whole numbers
            hebrew_num = self.heb2num(int(num), current_gender)
            hebrew_num = f"{clean_prefix}{hebrew_num}"

        if track_changes and number_changes is not None:
            number_changes[match.group(0)] = hebrew_num

        return f"{hebrew_num} {noun}" if '.' not in num else hebrew_num

    def _process_date_patterns(self, text):
        """Simplified date preprocessing using consolidated patterns"""
        import re

        # Use consolidated preprocessing patterns with validation
        for pattern, replacement in self._consolidated_date_patterns['preprocessing']:
            def safe_replace(match):
                # Apply validation for different pattern types
                if r'(\d{1,2})' in pattern and r'(\d{1,2})' in pattern:
                    # DD.MM patterns - validate day/month
                    groups = match.groups()
                    if len(groups) >= 3:
                        day, month = groups[-2], groups[-1]  # Last two numeric groups
                        if not self._safe_date_processor().validate_date_components(int(day), int(month)):
                            return match.group(0)
                elif self.get_consolidated_date_patterns()['yyyy_mm_dd'] in pattern:
                    # YYYY.MM.DD patterns - validate with year
                    year, month, day = match.group(1), match.group(2), match.group(3)
                    if not self._safe_date_processor().validate_date_components(int(day), int(month), int(year)):
                        return match.group(0)

                # Apply replacement if validation passed
                return re.sub(pattern, replacement, match.group(0))

            text = re.sub(pattern, safe_replace, text)

        return text

    def initialize_dataframe(self):
        if self.debug:
            print(f"[INIT_DEBUG] Starting initialize_dataframe...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            excel_path = os.path.join(script_dir, EXCEL_FILENAME)

            if self.debug:
                print(f"Loading Excel file: {excel_path}")

            df_main = pd.read_excel(excel_path, engine='openpyxl')
            df_main = df_main.fillna('')  # Ensure no column value is nan

            try:
                df_noun_genders = pd.read_excel(excel_path, sheet_name='Noun_Genders', engine='openpyxl')
                df_noun_genders = df_noun_genders.fillna('')

                if 'שם עצם' in df_noun_genders.columns and 'מין' in df_noun_genders.columns:
                    for _, row in df_noun_genders.iterrows():
                        noun = row['שם עצם']
                        gender = row['מין']
                        noun_type = row.get('סוג', '') if 'סוג' in df_noun_genders.columns else ''

                        if noun and gender:
                            noun = str(noun).replace("'", "׳")
                            if gender == 'זכר':  # masculine
                                self._hebrew_nouns_gender[noun] = 'm'
                            elif gender == 'נקבה':  # feminine
                                self._hebrew_nouns_gender[noun] = 'f'

                            if noun_type and str(noun_type).strip():
                                self._hebrew_nouns_types[noun] = str(noun_type).strip()

                    if self.debug:
                        print(f"Loaded {len([k for k in self._hebrew_nouns_gender if k not in ['שקלים', 'כוכבית', 'דקות', 'PSI', 'אגורות']])} noun genders from Noun_Genders sheet")
                        print(f"Total nouns in gender dictionary: {len(self._hebrew_nouns_gender)}")
                        print(f"Total nouns with type information: {len(self._hebrew_nouns_types)}")

            except Exception as e:
                print(f"Could not read Noun_Genders sheet: {str(e)}")

            column_mapping = {
                'מקור': 'Original',
                'ניקוד': 'Nikud',  # Changed from מנוקד to ניקוד to match Excel file
                'גוף': 'person_value',
                'זכר': 'Zachar',
                'נקבה': 'Nekeva'
            }
            df_main = df_main.rename(columns=column_mapping)

            # Add gender_source column to track where gender information comes from
            df_main['gender_source'] = ''

            if 'Original' in df_main.columns:
                # Vectorized operations for better performance
                df_main['Original'] = df_main['Original'].astype(str)
                df_main['Original'] = df_main['Original'].apply(lambda x: unicodedata.normalize('NFC', x)).str.replace("'", "׳")

            text_columns = ['Nikud', 'person_value', 'Zachar', 'Nekeva']
            for col in text_columns:
                if col in df_main.columns:
                    # Vectorized string operations for better performance
                    df_main[col] = df_main[col].fillna('').astype(str).str.replace("'", "׳")

            if 'אגורות' not in df_main['Original'].values:
                new_row = pd.DataFrame({
                    'Original': ['אגורות'],
                    'Nikud': ['אֲגוֹרוֹת'],
                    'person_value': [''],
                    'Zachar': [''],
                    'Nekeva': ['']
                })
                df_main = pd.concat([df_main, new_row], ignore_index=True)
                if self.debug:
                    print("Added missing 'אגורות' entry to dictionary")

            self._translation_df = df_main

            if self.debug:
                print(f"[INIT_DEBUG] Calling _create_optimized_tables...")
            self._create_optimized_tables(df_main)
            if self.debug:
                print(f"[INIT_DEBUG] Finished _create_optimized_tables")

            self._load_phonetic_gender_mappings()

            self._is_initialized = True
            if self.debug:
                print(f"Dictionary initialized with {len(df_main)} total entries")

        except Exception as e:
            print(f"Error loading dictionary: {str(e)}")
            self._translation_df = pd.DataFrame(columns=['Original', 'Nikud', 'person_value', 'Zachar', 'Nekeva'])
            self._is_initialized = False

    def _create_optimized_tables(self, df_main):
        """Create ultra-fast dictionary lookups + fallback sorted table"""
        try:
            # The Excel file is already preprocessed by load_milon.py with first words and Special column
            # No need for additional preprocessing - just use the data as-is

            # ULTRA-FAST: Create dictionary lookup (288x faster than pandas!)
            self._milon_dict = {}
            for _, row in df_main.iterrows():
                original = str(row.get('Original', '')).strip()
                if original:
                    # Normalize the key for consistent lookup
                    normalized_key = self._TextCleaningUtils.normalize_quotes(original).lower()
                    self._milon_dict[normalized_key] = row.to_dict()

            # Initialize search statistics for debug mode
            self._search_stats = {
                'dict_hits': 0,
                'dict_misses': 0,
                'pandas_searches': 0,
                'total_searches': 0
            }

            # Keep unified table for backward compatibility and complex searches
            # Data is already sorted by load_milon.py for optimal binary search performance
            self._milon_unified = df_main.copy()

            # Keep these for backward compatibility (all point to the same unified table)
            self._milon_1_word = self._milon_unified
            self._milon_2_word = self._milon_unified
            self._milon_3_word = self._milon_unified
            self._milon_4plus_word = self._milon_unified

            # Set optimized_tables for _pure_milon_lookup compatibility
            self.optimized_tables = {
                'unified': type('Table', (), {'data': self._milon_unified})()
            }

            if self.debug:
                print(f"ULTRA-FAST dictionary lookup created:")
                print(f"  Dictionary entries: {len(self._milon_dict)} entries (O(1) lookup)")
                print(f"  Fallback unified table: {len(self._milon_unified)} entries (pre-sorted)")
                print(f"  Pre-processed Excel file with Special entries and sorting loaded successfully!")

        except Exception as e:
            print(f"Error creating unified table: {str(e)}")
            self._milon_unified = df_main
            self._milon_1_word = df_main
            self._milon_2_word = df_main
            self._milon_3_word = df_main
            self._milon_4plus_word = df_main







    # OLD replace_with_punctuation method REMOVED - using table-based processing instead
    # Use process_text_with_table instead for all text processing
        # Method removed - use process_text_with_table() instead
        return "", {}

    def _print_search_statistics(self):
        """Print Excel search performance statistics (debug mode only)"""
        if not hasattr(self, '_search_stats'):
            return

        stats = self._search_stats
        total = stats['total_searches']
        hits = stats['dict_hits']
        misses = stats['dict_misses']
        pandas_used = stats['pandas_searches']

        if total == 0:
            return

        hit_rate = (hits / total * 100) if total > 0 else 0
        dict_efficiency = (hits / (hits + pandas_used) * 100) if (hits + pandas_used) > 0 else 0

        print("[STATS] EXCEL SEARCH PERFORMANCE STATISTICS:")
        print(f"  [TOTAL] Total searches: {total}")
        print(f"  [HITS] Dictionary hits: {hits} ({hit_rate:.1f}%)")
        print(f"  [MISS] Dictionary misses: {misses}")
        print(f"  [PANDAS] Pandas fallback: {pandas_used}")
        print(f"  [EFFICIENCY] Dictionary efficiency: {dict_efficiency:.1f}% (hits vs pandas)")
        if hits > 0:
            print(f"  💫 Speed improvement: ~{hits * 287:.0f}ms saved vs binary search!")

    # merge_vav_number function removed - not needed since input text doesn't contain nikud

    def apply_vav_shuruk(self, word):
        """Apply VAV shuruk rule: וְ + sheva → וּ"""
        import re

        # Original specific cases
        if word.startswith('ו'):
            for target in ['שניים', 'שלושה', 'שמונה']:
                if word[1:] == target:
                    return 'וּ' + target

        # General rule: וְ followed by character with sheva → וּ
        # Pattern: וְ (VAV + SHEVA) followed by any character that also has SHEVA
        # Hebrew SHEVA is Unicode U+05B0 (◌ְ)
        pattern = r'וְ([א-ת]ְ)'  # VAV+SHEVA followed by Hebrew letter+SHEVA

        def replace_vav_sheva(match):
            # Replace וְ with וּ, keep the rest unchanged
            return 'וּ' + match.group(1)

        result = re.sub(pattern, replace_vav_sheva, word)
        return result

    def _finalize_number_result(self, result, is_year=False):
        """Apply vav shuruk and add comma for years"""
        original_result = result
        result = self.apply_vav_shuruk(result)
        if is_year:
            result += ','

        # Debug print result
        if self.debug:
            print(f"[HEB2NUM_OUTPUT] original='{original_result}' → final='{result}' (is_year={is_year})")
            if self.debug_file:
                self.debug_file.write(f"[HEB2NUM_OUTPUT] original='{original_result}' → final='{result}' (is_year={is_year})\n")
                self.debug_file.flush()

        return result

    def heb2num(self, n, g='m', debug=False, construct_state=False, is_year=False):
        # Debug print input parameters
        self._debug_print(f"[HEB2NUM_INPUT] n={n}, g='{g}', debug={debug}, construct_state={construct_state}, is_year={is_year}")

        if (11000 <= n <= 19000):
            thousands = n // 1000
            remainder = n % 1000
            teen_idx = thousands - 10
            teen_form = HEBREW_TEENS['m'][teen_idx]

            if remainder == 0:
                return self._finalize_number_result(f"{teen_form} {HEBREW_THOUSAND}", is_year)

            remainder_text = self.heb2num(remainder, g)
            return self._finalize_number_result(f"{teen_form} {HEBREW_THOUSAND} {remainder_text}", is_year)

        if n == 0:
            self._debug_print(f"[HEB2NUM_OUTPUT] Zero case: n=0 → 'אֶפֶס'")
            return 'אֶפֶס'

        if n < 0:
            return self._finalize_number_result(f"מִינוּס {self.heb2num(abs(n), g)}", is_year)

        num_str = str(n)
        length = len(num_str)

        if length == 1:
            if construct_state:
                # Check construct state forms for numbers 1-2 (gender-specific)
                if n in HEBREW_CONSTRUCT_UNITS_LOW[g]:
                    result = HEBREW_CONSTRUCT_UNITS_LOW[g][n]
                # Check construct state forms for numbers 3-9 (gender-neutral)
                elif n in HEBREW_CONSTRUCT_UNITS:
                    result = HEBREW_CONSTRUCT_UNITS[n]
                else:
                    result = HEBREW_UNITS[g][n]
            else:
                result = HEBREW_UNITS[g][n]
            return self._finalize_number_result(result, is_year)

        if length == 2 and 10 <= n <= 19:
            # Special case for 10 with construct state
            if n == 10 and construct_state and n in HEBREW_CONSTANTS['CONSTRUCT_UNITS']:
                result = HEBREW_CONSTANTS['CONSTRUCT_UNITS'][n]
            else:
                result = HEBREW_TEENS[g][n - 10]
            return self._finalize_number_result(result, is_year)

        if length == 2:
            if n % 10 == 0:  # Exact tens
                result = HEBREW_TENS[n // 10]
                return self._finalize_number_result(result, is_year)
            unit_part = HEBREW_UNITS[g][n % 10]
            if n % 10 in [2, 8]:
                vav = "וּ"
            else:
                vav = "וְ"
            result = f"{HEBREW_TENS[n // 10]} {vav}{unit_part}"
            return self._finalize_number_result(result, is_year)

        if length == 3:
            if n % 100 == 0:  # Exact hundreds
                result = HEBREW_HUNDREDS[n // 100]
                return self._finalize_number_result(result, is_year)
            result = f"{HEBREW_HUNDREDS[n // 100]} {self.heb2num(n % 100, g)}"
            return self._finalize_number_result(result, is_year)

        if n >= 1000:
            thousands = n // 1000
            remainder = n % 1000
            if thousands >= 11:
                thousands_text = self.heb2num(thousands, 'm')
                if remainder == 0:
                    result = f"{thousands_text} {HEBREW_THOUSAND}"
                    return self._finalize_number_result(result, is_year)
                remainder_text = self.heb2num(remainder, g)
                result = f"{thousands_text} {HEBREW_THOUSAND} {remainder_text}"
                return self._finalize_number_result(result, is_year)
            if thousands == 1:
                prefix = HEBREW_THOUSAND
            elif thousands == 2:
                prefix = 'אַלְפַּיִם'
            elif 3 <= thousands <= 10:
                if thousands in HEBREW_CONSTRUCT_UNITS:
                    prefix = f"{HEBREW_CONSTRUCT_UNITS[thousands]} {HEBREW_THOUSANDS}"
                else:
                    if thousands == 10:
                        prefix = f"עֲשֶׂרֶת {HEBREW_THOUSANDS}"
                    else:
                        prefix = f"{HEBREW_UNITS['m'][thousands]} {HEBREW_THOUSANDS}"
            elif 11 <= thousands <= 19:
                teen_form = HEBREW_TEENS['m'][thousands - 10]
                prefix = f"{teen_form} {HEBREW_THOUSAND}"
            else:
                prefix = f"{self.heb2num(thousands, 'm')} {HEBREW_THOUSANDS}"

            if remainder == 0:
                return self._finalize_number_result(prefix, is_year)
            result = f"{prefix} {self.heb2num(remainder, g)}"
            return self._finalize_number_result(result, is_year)

        if length >= 5:
            if length <= 6:
                ten_thousands = n // 1000
                remainder = n % 1000

                if remainder == 0 and ten_thousands % 1000 == 0:
                    thousands_text = self.heb2num(ten_thousands // 1000, 'm')
                    if ten_thousands // 1000 == 1:
                        result = "אֶלֶף אֶלֶף"  # One thousand thousand
                        return self._finalize_number_result(result, is_year)
                    result = f"{thousands_text} אֶלֶף"
                    return self._finalize_number_result(result, is_year)

                if 11 <= ten_thousands <= 19:
                    teen_form = HEBREW_TEENS['m'][ten_thousands - 10]
                    thousands_text = f"{teen_form} אֶלֶף"
                else:
                    thousands_text = self.heb2num(ten_thousands, 'm')
                    thousands_text = f"{thousands_text} אֲלָפִים"

                if remainder == 0:
                    return self._finalize_number_result(thousands_text, is_year)

                remainder_text = self.heb2num(remainder, g)
                result = f"{thousands_text} {remainder_text}"
                return self._finalize_number_result(result, is_year)

        result = f"מספר {n}"
        return self._finalize_number_result(result, is_year)

    def convert_numbers_to_hebrew(self, text, number_cache=None, gender=None, track_changes=False):
        """Convert numbers to Hebrew text without creating SSML markup"""
        if number_cache is None:
            number_cache = {}

        processed_text = text
        target_tts_gender = "female" if gender == 'f' else "male"

        number_changes = {}

        try:
            processed_text, number_first_changes = self.process_numbers_with_noun_context(processed_text, track_changes=track_changes)
            if number_first_changes and track_changes:
                print(f"[NUMBER-FIRST] Applied {len(number_first_changes)} changes: {number_first_changes}")

                if number_first_changes:
                    if track_changes:
                        number_changes.update(number_first_changes)
                        return processed_text, number_changes
                    else:
                        return processed_text
        except Exception as e:
            print(f"[NUMBER-FIRST] Error: {e}, falling back to original approach")

        import re
        original_text = processed_text
        processed_text = re.sub(r'(\d+(?:,\d+)+(?:\.\d+)?)', lambda m: m.group(1).replace(',', ''), processed_text)

        if original_text != processed_text and track_changes:
            print(f"[COMMA REMOVAL] Original: {original_text}")
            print(f"[COMMA REMOVAL] After:    {processed_text}")

        currency_nouns = self._get_currency_nouns()
        currency_pattern = '|'.join(currency_nouns)
        decimal_number_noun_pattern = rf'([א-ת])?(\d+)\.(\d+)\s+({currency_pattern})'
        percentage_pattern = r'(\d+(?:\.\d+)?)%'  # Match percentage: 12% or 12.5%
        decimal_number_pattern = r'([א-ת])?(\d+)\.(\d+)'  # Match decimal number with optional Hebrew prefix
        hyphen_range_pattern = r'(\d{1,3}(?:,\d{3})*)-(\d{1,3}(?:,\d{3})*)\s+([א-ת]+)'
        range_ad_pattern = r'(ל[-־])(\d+)\s+עד\s+(\d+)\s+([א-ת]+)'
        kochavit_pattern = r'(כוכבית)\s+(\d+)'
        prefix_comma_number_pattern = r'([בהלמשו])[-־](\d{1,3}(?:,\s*\d{3})+)'
        prefix_number_pattern = r'([בהלמשו])[-־](\d+)'
        comma_number_shekel_pattern = rf'(\d{{1,3}}(?:,\s*\d{{3}})+)\s+({currency_pattern})'
        regular_number_noun_pattern = r'([א-ת])?(\d+)\s+([א-ת]+)'

        def handle_currency_decimal(prefix, whole_str, decimal_str, currency_noun):
            currency_gender = self._hebrew_nouns_gender.get(currency_noun, 'm')  # Default to masculine if not found

            whole_text = self.heb2num(int(whole_str), currency_gender, construct_state=True)

            clean_prefix = prefix.rstrip('-') if prefix else ''

            is_israeli_currency = self._hebrew_nouns_types.get(currency_noun, '') == 'שקלים'
            if is_israeli_currency:
                decimal_normalized = decimal_str.ljust(2, '0') if len(decimal_str) == 1 else decimal_str
                decimal_value = int(decimal_normalized)

                # Special case for 1 shekel: use singular "שקל אחד" instead of "אחד שקלים"
                if whole_str == "1":
                    if decimal_value == 0:
                        return f"{clean_prefix}שקל אחד"
                    elif decimal_value == 1:
                        # 1.01 שקלים -> שקל אחד ואגורה אחת
                        return f"{clean_prefix}שקל אחד ואגורה אחת"
                    elif decimal_value == 2:
                        # 1.02 שקלים -> שקל אחד ושתי אגורות
                        return f"{clean_prefix}שקל אחד ושתי אגורות"
                    else:
                        decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                        hebrew_result = f"{clean_prefix}שקל אחד ו{decimal_text} אגורות"
                        print(f"[DEBUG] HANDLE_CURRENCY_DECIMAL: {whole_str}.{decimal_str} {currency_noun} -> '{hebrew_result}' (1 shekel singular)")
                        return hebrew_result
                else:
                    if decimal_value == 0:
                        return f"{clean_prefix}{whole_text} שקלים"
                    elif decimal_value == 1:
                        # Special case for 1 agora: "אגורה אחת" instead of "אחת אגורות"
                        return f"{clean_prefix}{whole_text} שקלים ואגורה אחת"
                    elif decimal_value == 2:
                        # Special case for 2 agorot: "שתי אגורות" instead of "שתיים אגורות"
                        return f"{clean_prefix}{whole_text} שקלים ושתי אגורות"

                    decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                    hebrew_result = f"{clean_prefix}{whole_text} שקלים ו{decimal_text} אגורות"
                    print(f"[DEBUG] HANDLE_CURRENCY_DECIMAL: {whole_str}.{decimal_str} {currency_noun} -> '{hebrew_result}'")
                    return hebrew_result
            else:
                decimal_value = int(decimal_str)
                if decimal_value == 0:
                    return f"{clean_prefix}{whole_text} {currency_noun}"
                else:
                    decimal_text = self.heb2num(decimal_value, currency_gender)
                    return f"{clean_prefix}{whole_text} נקודה {decimal_text} {currency_noun}"

        def replace_decimal_number_noun(match):
            if match.group(24) and match.group(25) and match.group(26) and match.group(27):  # With prefix
                prefix = match.group(24)
                whole = match.group(25)
                decimal = match.group(26)
                noun = match.group(27)
            elif match.group(25) and match.group(26) and match.group(27):  # Without prefix
                prefix = ''
                whole = match.group(25)
                decimal = match.group(26)
                noun = match.group(27)
            else:
                return match.group(0)

            current_gender = self._hebrew_nouns_gender.get(noun, 'f')  # Default to feminine when gender is unknown
            tts_gender = "female" if current_gender == 'f' else "male"

            is_israeli_currency = self._hebrew_nouns_types.get(noun, '') == 'שקלים'
            if is_israeli_currency:
                hebrew_num = handle_currency_decimal(prefix, whole, decimal, noun)
            else:
                whole_text = self.heb2num(int(whole), current_gender)
                decimal_text = self.heb2num(int(decimal), current_gender)

                hebrew_num = f"{prefix}{whole_text} {noun} נקודה {decimal_text}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_num

            print(f"[DEBUG] CURRENCY DECIMAL REPLACEMENT: '{prefix}{whole}.{decimal} {noun}' -> '{hebrew_num}' (noun: {noun})")

            # Store directly in table if available
            # Note: table is from parent function scope - process_numbers_with_noun_context
            if table:  # type: ignore
                original = f"{prefix}{whole}.{decimal} {noun}"
                self._store_conversion_in_table(table, original, hebrew_num, "CURRENCY_DECIMAL")  # type: ignore

            return hebrew_num

        def replace_decimal_number(match):
            if match.group(28) and match.group(29) and match.group(30):  # With prefix
                prefix = match.group(28)
                whole = match.group(29)
                decimal = match.group(30)
            elif match.group(29) and match.group(30):  # Without prefix
                prefix = ''
                whole = match.group(29)
                decimal = match.group(30)
            else:
                return match.group(0)

            try:
                day = int(whole)
                month = int(decimal)

                if (1 <= day <= 31 and 1 <= month <= 12 and
                    prefix and prefix.endswith('-')):
                    return match.group(0)
            except ValueError:
                pass  # Not valid integers, continue with decimal processing

            tts_gender = "female"  # Default to feminine when gender is unknown

            clean_prefix = prefix.rstrip('-') if prefix else ''

            whole_text = self.heb2num(int(whole), 'f')  # Default to feminine when gender is unknown
            decimal_text = self.heb2num(int(decimal), 'f')

            hebrew_num = f"{clean_prefix}{whole_text} נקודה {decimal_text}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_num

            return hebrew_num

        def replace_regular_number_noun(match):
            prefix = match.group(27) if match.group(27) and match.group(28) else ''
            num_noun = match.group(28) if match.group(28) else None

            if not num_noun:
                return match.group(0)

            parts = num_noun.strip().split()
            if len(parts) != 2:
                return match.group(0)

            num, noun = parts[0], parts[1]
            return self._handle_number_noun_pattern(match, prefix, num, noun, track_changes, number_changes)

        def find_gender_context(match_start, match_end):
            """
            Find gender context by looking forward up to 3 words or backward up to 4 words.
            Returns tuple: (gender, tts_gender)
            """
            number_text = processed_text[match_start:match_end].strip()
            number_match = re.search(r'\d+', number_text)
            if number_match:
                number_str = number_match.group(0)
                if len(number_str) == 4 and number_str.isdigit():
                    year_num = int(number_str)
                    if 1900 <= year_num <= 2100:  # Valid year range
                        return 'f', "female"  # Always feminine for years

            remaining_text = processed_text[match_end:].strip()

            next_char_match = re.match(r'^(\s*)(.)', remaining_text)
            if next_char_match:
                next_char = next_char_match.group(2)
                if next_char in self._hebrew_nouns_gender:
                    current_gender = self._hebrew_nouns_gender[next_char]
                    return current_gender, "female" if current_gender == 'f' else "male"

            words_after = remaining_text.split()
            for i in range(min(3, len(words_after))):
                word = words_after[i]

                clean_word = word  # Remove  markers
                clean_word = re.sub(r'[^\u05D0-\u05EA]', '', clean_word)  # Keep only Hebrew letters
                clean_word = clean_word.strip()

                if clean_word and clean_word in self._hebrew_nouns_gender:
                    current_gender = self._hebrew_nouns_gender[clean_word]
                    return current_gender, "female" if current_gender == 'f' else "male"

            text_before = processed_text[:match_start]
            words_before = text_before.split()

            for i in range(min(4, len(words_before))):
                word = words_before[-(i+1)]  # Start from the last word and go backward

                clean_word = word  # Remove  markers
                clean_word = re.sub(r'[^\u05D0-\u05EA]', '', clean_word)  # Keep only Hebrew letters
                clean_word = clean_word.strip()

                if clean_word and clean_word in self._hebrew_nouns_gender:
                    current_gender = self._hebrew_nouns_gender[clean_word]
                    return current_gender, "female" if current_gender == 'f' else "male"

            return 'f', "female"

        def replace_number(match):
            if match.group(29) and match.group(30):  # With prefix
                prefix = match.group(29)
                num = match.group(30)
            elif match.group(30):  # Without prefix
                prefix = ''
                num = match.group(30)
            else:
                return match.group(0)

            match_start = match.start()
            match_end = match.end()
            current_gender, tts_gender = find_gender_context(match_start, match_end)

            if '.' in num:
                clean_prefix = prefix.rstrip('-') if prefix else ''
                whole_part, decimal_part = num.split('.')
                whole_text = self.heb2num(int(whole_part), current_gender)
                decimal_text = self.heb2num(int(decimal_part), 'f')

                hebrew_num = f"{clean_prefix}{whole_text} נקודה {decimal_text}"

                if track_changes:
                    original = match.group(0)
                    number_changes[original] = hebrew_num

                return hebrew_num
            else:
                clean_prefix = prefix.rstrip('-') if prefix else ''
                hebrew_num = self.heb2num(int(num), current_gender)

                if track_changes:
                    original = f"{prefix}{num}"
                    replacement = f"{clean_prefix}{hebrew_num}"
                    number_changes[original] = replacement

                return f"{clean_prefix}{hebrew_num}"

        hebrew_text_pattern = r'[א-ת־]+'  # Hebrew letters + hyphens (no  markers)

        hebrew_months_list = list(self._hebrew_months.values())
        months_with_b = [f'ב{month}' for month in hebrew_months_list]
        all_month_patterns = months_with_b + hebrew_months_list

        hebrew_months_pattern = '|'.join(all_month_patterns)

        # Get centralized date patterns to avoid duplication
        date_patterns = self.get_consolidated_date_patterns()

        pattern = re.compile(
            rf'(ריבית|רבית)\s+של\s+(\d+(?:[.\/-]\d+)?)%(?!\d)'      # interest rate pattern: "ריבית של 12.5%" - HIGHEST PRIORITY (supports ., /, -)
            rf'|(\d+(?:[.\/-]\d+)?)%\s+(ריבית|רבית)'                # percentage before interest: "12.5% ריבית" - HIGHEST PRIORITY (supports ., /, -)
            rf'|({hebrew_text_pattern})\s+(\d+(?:[.\/-]\d+)?)%(?!\d)' # Hebrew word + percentage: "הריבית היא 12.5%" - HIGHEST PRIORITY (supports ., /, -)
            rf'|(\d+(?:[.\/-]\d+)?)%(?!\d)'                           # percentage pattern: "12%" or "12.5%" - HIGHEST PRIORITY (supports ., /, -)
            rf'|([א-ת]-)?(\d+)\s+ו?עד\s+(\d+)\s+({hebrew_text_pattern})'  # range pattern: "מ-3 עד 18 תשלומים" - VERY HIGH PRIORITY (before other number patterns)
            rf'|{date_patterns["mm_yyyy"]}'                                     # MM.YYYY date pattern: "01.2025" - HIGH PRIORITY (before decimal patterns)
            rf'|(\d+)\s+({hebrew_months_pattern})\s+(\d{{4}})'        # DD MONTH YYYY (e.g., 16 דצמבר 2024)
            rf'|(\d+)\s+({hebrew_months_pattern})'                    # DD MONTH (without year, e.g., 16 דצמבר)
            rf'|({hebrew_months_pattern})\s+(\d{{4}})'                # MONTH YYYY (e.g., לפברואר 2025, נובמבר 2024)
            rf'|{date_patterns["yyyy_mm_dd"]}'                   # YYYY.MM.DD format (e.g., 2025.03.05) - Year.Month.Day with dots (HIGHEST PRIORITY)
            rf'|{date_patterns["dd_mm_yyyy"]}'                   # DD.MM.YYYY format (e.g., 07.03.2025) - Day.Month.Year with dots
            rf'|(?<!\d){date_patterns["dd_mm"]}(?!\d)'                # DD.MM format (e.g., 5.4) - Day.Month with dots (with word boundaries)
            rf'|([א-ת]+[-־]?)?(\d+)\.(\d+)\s+({currency_pattern})'      # decimal number + currency with optional prefix
            rf'|([א-ת]+[-־]?)?(\d+)\.(\d+)'                             # decimal number with optional prefix
            rf'|([א-ת]-)?(\d+)\s+([א-ת]+)'                           # simple prefix pattern: "ב-5 תשלומים"
            rf'|([א-ת]+[-־]?)?(\d+\s+[א-ת]+)'                           # number + noun with optional prefix
            rf'|([א-ת]+[-־]?)?(\d+)'                                    # number with optional prefix
        )

        def replace_mm_yyyy_date(match):
            """Handle MM.YYYY date patterns: 01.2025, 12.2024, etc.
            Convert to Hebrew: MONTH אלפיים עשרים וחמש
            """
            month_str = match.group(12)  # '01'
            year_str = match.group(13)   # '2025'

            try:
                month_num = int(month_str)
                year_num = int(year_str)

                if 1 <= month_num <= 12 and 1900 <= year_num <= 2100:
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)

                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    hebrew_text = f"{month_name} {year_text}"

                    if track_changes:
                        original = match.group(0)
                        number_changes[original] = hebrew_text

                    return hebrew_text
            except (ValueError, KeyError):
                pass  # Fall through to normal processing if parsing fails

            return match.group(0)

        def replace_dd_mm_yyyy_date(match):
            """Handle DD-MM-YYYY date patterns: 07-03-2025, 15-12-2024, etc.
            Convert to Hebrew: שביעי בMONTH אלפיים עשרים וחמש
            """
            day_str = match.group(21)    # '07'
            month_str = match.group(22)  # '03'
            year_str = match.group(23)   # '2025'

            try:
                day_num = int(day_str)
                month_num = int(month_str)
                year_num = int(year_str)

                if self._safe_date_processor().validate_date_components(day_num, month_num, year_num):
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                    hebrew_text = self._safe_date_processor().format_hebrew_date(day_text, month_name, year_text)

                    if track_changes:
                        original = match.group(0)
                        number_changes[original] = hebrew_text

                    return hebrew_text
            except (ValueError, KeyError):
                pass  # Fall through to normal processing if parsing fails

            return match.group(0)

        def replace_date_month(match):
            """Handle number + month patterns: 10 ביוני, 15 בנובמבר, etc.
            Rule: 1-10 → ordinals (ראשון, שני, ... עשירי)
                  11+ → cardinal male numbers (אחד עשר, שנים עשר, ...)
            """
            num = match.group(16)          # '10'
            month_with_prefix = match.group(17)  # 'ביוני' or 'יוני'

            num_value = int(num)

            hebrew_num = self._safe_date_processor().convert_day_to_hebrew(num_value)

            hebrew_text = f"{hebrew_num} {month_with_prefix}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_text

            return hebrew_text

        def replace_word_date_month(match):
            """Handle word + number + month patterns: תאריך 10 בדצמבר, יום 5 באפריל, etc.
            Rule: 1-10 → ordinals (ראשון, שני, ... עשירי)
                  11+ → cardinal male numbers (אחד עשר, שנים עשר, ...)
            """
            word = match.group(14)             # 'תאריך'
            num = match.group(15)              # '10'
            month_with_prefix = match.group(16)  # 'בדצמבר' or 'דצמבר'

            num_value = int(num)

            hebrew_num = self._safe_date_processor().convert_day_to_hebrew(num_value)

            hebrew_text = f"{word} {hebrew_num} {month_with_prefix}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_text

            return hebrew_text

        def replace_range_ad(match):
            prefix = match.group(8) if match.group(8) else ''  # 'מ-' or empty
            first_num = match.group(9)  # '3'
            second_num = match.group(10)  # '18'
            noun = match.group(11)  # 'תשלומים'

            clean_noun = self._TextCleaningUtils.clean_noun_for_lookup(noun)

            current_gender = self._hebrew_nouns_gender.get(clean_noun, 'f')  # Default to feminine when gender is unknown
            tts_gender = "female" if current_gender == 'f' else "male"

            first_hebrew = self.heb2num(int(first_num), current_gender)
            second_hebrew = self.heb2num(int(second_num), current_gender)

            clean_prefix = prefix.rstrip('-') if prefix else ''

            hebrew_range = f"{clean_prefix}{first_hebrew} עד {second_hebrew} {noun}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_range

            return hebrew_range

        def replace_simple_prefix(match):
            prefix = match.group(24)  # 'ב-', 'ל-', 'מ-', etc.
            num = match.group(25)     # '5'
            noun = match.group(26)    # 'תשלומים'

            clean_noun = self._TextCleaningUtils.clean_noun_for_lookup(noun)

            current_gender = self._hebrew_nouns_gender.get(clean_noun, 'f')  # Default to feminine when gender is unknown
            tts_gender = "female" if current_gender == 'f' else "male"

            # NEW RULE: Check if noun starts with ה (definite article) to determine construct state
            # If noun starts with ה (like התשלומים), use construct state (חֲמֵשֶׁת)
            # If no ה (like תשלומים), use absolute state (חֲמִישָׁה)
            use_construct_state = noun.startswith('ה')

            hebrew_num = self.heb2num(int(num), current_gender, construct_state=use_construct_state)

            clean_prefix = prefix.rstrip('-') if prefix else ''

            hebrew_text = f"{clean_prefix}{hebrew_num} {noun}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_text

            return hebrew_text

        def replace_hebrew_word_number(match):
            hebrew_word = match.group(29)  # 'מוקד' or 'עד'
            num = match.group(30)          # '103', '12', etc.

            match_start = match.start()
            match_end = match.end()
            current_gender, tts_gender = find_gender_context(match_start, match_end)

            hebrew_num = self.heb2num(int(num), current_gender)

            hebrew_text = f"{hebrew_word} {hebrew_num}"

            if track_changes:
                original = match.group(0)
                number_changes[original] = hebrew_text

            return hebrew_text

        def unified_replace(match):
            return self._unified_number_replace(match, track_changes, number_changes,
                                              replace_range_ad, replace_mm_yyyy_date, replace_word_date_month,
                                              replace_date_month, replace_dd_mm_yyyy_date, replace_simple_prefix,
                                              replace_decimal_number_noun, replace_decimal_number,
                                              replace_hebrew_word_number, replace_regular_number_noun, replace_number)

        def should_exclude_match(match):
            full_match = match.group(0)
            start_pos = match.start()
            end_pos = match.end()

            text_before_match = processed_text[:start_pos]
            text_after_match = processed_text[end_pos:]

            last_opening_tilde = text_before_match.rfind('')
            next_closing_tilde = text_after_match.find('')

            if last_opening_tilde != -1 and next_closing_tilde != -1:
                text_between_tilde_and_match = text_before_match[last_opening_tilde+1:]
                if '' not in text_between_tilde_and_match:
                    return True

            context_before = processed_text[max(0, start_pos-10):start_pos]
            context_after = processed_text[end_pos:end_pos+10]
            full_context = context_before + full_match + context_after

            if re.search(r'\d{1,2}:\d{2}', full_context):
                return True

            if 'בין' in context_before and ':' in context_after:
                return True

            if re.match(r':\d{2}', context_after):
                return True

            number_match = re.search(r'\d+', full_match)
            if number_match:
                number_str = number_match.group(0)
                if len(number_str) >= 7:
                    return True

            return False

        def safe_unified_replace(match):
            if should_exclude_match(match):
                return match.group(0)  # Return original, don't convert
            return unified_replace(match)

        processed_text = pattern.sub(safe_unified_replace, processed_text)

        if track_changes:
            return processed_text, number_changes
        return processed_text

    def extract_clean_text(self, text, remove_markers=True):
        """Extract clean text from processed text, optionally removing markers"""
        text = re.sub(r'\s+', ' ', text)

        text = text

        if remove_markers:
            text = text

        return text.strip()

    def _remove_urls(self, text):
        text = re.sub(r'בקישור המצורף\s+https?://[^\s]+', 'בקישור המצורף.', text)

        text = re.sub(r'(?:,|\.|:|\s+)https?://[^\s]+', '.', text)

        text = re.sub(r'https?://[^\s]+', '', text)

        return text

    def normalize_hebrew_double_quote(self, text):
        return text.replace('״', '"')

    def process_text(self, text):
        import time
        start_time = time.perf_counter()

        if not text:
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            print(f"[RUNTIME] PROCESS_TEXT RUNTIME: {runtime_ms:.3f}ms (empty input)")
            return {
                'ssml_text': '',
                'tts_text': '',
                'show_text': '',
                'dict_words': {},
                'original_input': ''
            }

        if ("|" in text and not re.search(r'<speak>.*?</speak>', text, re.DOTALL)):
            return {
                'ssml_text': text,
                'ssml_clean': text,
                'ssml_marked': text,
                'show_text': text,
                'dict_words': {},
                'original_input': text,
                'line_by_line': {
                    'input_lines': [text],
                    'show_text_lines': [text],
                    'ssml_clean_lines': [text],
                    'ssml_marked_lines': [text],
                    'processed_lines': [text]
                }
            }

        clean_input_text = text
        self.ssml_input_emotions = None

        if re.search(r'<speak>.*?</speak>', text, re.DOTALL):
            speak_match = re.search(r'<speak>(.*?)</speak>', text, re.DOTALL)
            if speak_match:
                speak_content = speak_match.group(1).strip()

                emotion_match = re.search(r'<say-as>\s*<([^>]+)>', speak_content, re.DOTALL)
                if emotion_match:
                    self.ssml_input_emotions = emotion_match.group(1)
                    if self.debug:
                        self._debug(f"Extracted SSML emotion: '{self.ssml_input_emotions}'")

                say_as_match = re.search(r'(.*?)\s*<say-as>', speak_content, re.DOTALL)
                if say_as_match:
                    clean_input_text = say_as_match.group(1).strip()
                else:
                    clean_input_text = speak_content

        clean_input_text = self.normalize_hebrew_double_quote(clean_input_text)

        clean_input_text = clean_input_text.replace('־', '-')  # Normalize Hebrew dashes
        clean_input_text = clean_input_text.replace('—', ' ')  # Replace em dash with space

        original_input = clean_input_text
        if self.debug:
            self._debug(f"Original input text: '{original_input}'")

        input_lines = clean_input_text.split('\n')

        processed_lines = []
        show_text_lines = []
        ssml_clean_lines = []
        raw_ssml_lines = []  # Store raw SSML with say-as tags
        all_dict_words = {}

        for line_num, line in enumerate(input_lines):
            if not line.strip():  # Skip empty lines
                processed_lines.append('')
                show_text_lines.append('')
                ssml_clean_lines.append('')
                continue

            # Use the NEW table-based processing instead of old activate method
            # Pass line number for unique hamara table filenames
            line_number = f"_{line_num+1:02d}"  # Zero-padded (01, 02, 03, etc.)
            table_result = self.process_text_with_table(line, line_number=line_number)
            if table_result and 'ssml_clean' in table_result:
                activate_text_result = table_result['ssml_clean']
                dict_words = table_result.get('dict_words', {})
            else:
                activate_text_result, dict_words = '', {}

            all_dict_words.update(dict_words)

            show_text_lines.append(line)

            hebrew_months_pattern = '|'.join(self._hebrew_months.values())

            activate_text_result = self._process_date_patterns(activate_text_result)

            # Use consolidated date conversion patterns (combines centralized + Hebrew month patterns)
            centralized_patterns = self.get_consolidated_date_patterns()
            hebrew_month_patterns = self._consolidated_date_patterns['conversion']

            # Combine numeric patterns from centralized + Hebrew month patterns from compiled
            date_patterns = (
                [centralized_patterns['yyyy_mm_dd'], centralized_patterns['dd_mm_yyyy'],
                 centralized_patterns['dd_mm'], centralized_patterns['mm_yyyy'],
                 centralized_patterns['hebrew_prefix_dd_mm'], centralized_patterns['hebrew_prefix_dd_mm_yyyy']] +
                hebrew_month_patterns
            )

            def convert_date(match):
                # Simple fallback - return the match as-is for now
                return match.group(0)

            clean_text_input = self.extract_clean_text(activate_text_result, remove_markers=False)

            tts_text_input = clean_text_input
            for pattern in date_patterns:
                tts_text_input = re.sub(pattern, convert_date, tts_text_input)

            tts_text_input = self._remove_urls(tts_text_input)

            number_result = self.convert_numbers_to_hebrew(tts_text_input, gender=self.gender[-1], track_changes=True)
            if isinstance(number_result, tuple):
                processed_text_line, number_changes = number_result
                all_dict_words.update(number_changes)
            else:
                processed_text_line = number_result

            processed_lines.append(processed_text_line)

            clean_text_line = self._TextCleaningUtils.clean_punctuation_basic(processed_text_line)

            text_line_with_markers = processed_text_line

            original_words = line.split()
            clean_final_text = clean_text_line
            final_words = clean_final_text.split()

            self._track_actual_changes(original_words, final_words, all_dict_words)

            ssml_clean_lines.append(text_line_with_markers)

        def clean_commadot(text):
            return text.replace(',.', '.') if text else text

        # Clean all ssml_clean_lines ONCE here, then use the cleaned versions everywhere
        ssml_clean_lines_final = []
        for line in ssml_clean_lines:
            clean_line = line
            clean_line = clean_line  # Single point of TTS cleaning
            ssml_clean_lines_final.append(clean_line)

        show_text = '\n'.join(show_text_lines)  # This is now the original input text
        ssml_clean_with_markers = '\n'.join(ssml_clean_lines)

        ssml_clean = '\n'.join(ssml_clean_lines_final)  # Use pre-cleaned lines

        cleaned_original_input = clean_commadot(original_input)
        cleaned_show_text = clean_commadot(show_text)

        output_json = {
            "original_input": cleaned_original_input or '',
            "show_text": cleaned_show_text or '',
            "ssml_clean": ssml_clean or '',
            "line_by_line": {
                "input_lines": input_lines,
                "show_text_lines": show_text_lines,
                "ssml_clean_lines": ssml_clean_lines_final,  # Already cleaned
                "ssml_marked_lines": ssml_clean_lines,  # Keep markers as-is for proper highlighting
                "processed_lines": processed_lines
            }
        }
        with open("output_buffers.json", "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        clean_hebrew_text = ssml_clean

        emotion_tag = self.ssml_input_emotions if self.ssml_input_emotions else None

        if emotion_tag:
            final_ssml = f"<speak>\n{original_input}\n<say-as>\n<{emotion_tag}>{clean_hebrew_text}</>\n</say-as>\n</speak>"
        else:
            final_ssml = f"<speak>\n{original_input}\n<say-as>\n{clean_hebrew_text}\n</say-as>\n</speak>"

        print(f"[FINAL SSML FOR DISPLAY] {final_ssml}")

        tts_content_lines = []
        for line in processed_lines:
            if line.strip():
                # processed_lines items correspond to ssml_clean_lines_final by index
                line_index = processed_lines.index(line)
                if line_index < len(ssml_clean_lines_final):
                    clean_content = ssml_clean_lines_final[line_index]  # Use pre-cleaned content
                else:
                    clean_content = line.strip()
                    clean_content = clean_content  # Fallback cleaning
                tts_content_lines.append(clean_content)
            else:
                tts_content_lines.append(line)

        tts_ssml = f"<speak>\n{chr(10).join(tts_content_lines)}\n</speak>"
        print(f"[TTS SSML WITH CONVERTED HEBREW] {tts_ssml}")

        if self.debug:
            print('FINAL SSML (Line-by-Line):')
            print(final_ssml)
            print('\n=== TTS TEXT (Line-by-Line Debug) ===')
            for i, (input_line, ssml_line) in enumerate(zip(input_lines, ssml_clean_lines_final)):
                if input_line.strip():  # Only process lines that had input content
                    # ssml_line is already cleaned (from ssml_clean_lines_final)
                    formatted_tts_line = self._format_tts_for_debug(ssml_line)
                    print(formatted_tts_line)
                    print()  # Add empty line between processed lines

        end_time = time.perf_counter()
        runtime_ms = (end_time - start_time) * 1000
        print(f"[RUNTIME] PROCESS_TEXT RUNTIME: {runtime_ms:.3f}ms")
        print(f"🔊 FULL SSML BUFFER (with <speak> tags): {final_ssml}")

        return {
            'ssml_text': final_ssml,
            'ssml_clean': ssml_clean or '',
            'ssml_marked': ssml_clean_with_markers or '',  # Keep markers as-is for proper highlighting
            'show_text': cleaned_show_text or '',
            'dict_words': all_dict_words or {},
            'original_input': original_input or '',
            'line_by_line': {
                'input_lines': input_lines,
                'show_text_lines': show_text_lines,
                'ssml_clean_lines': ssml_clean_lines_final,  # Already cleaned
                'ssml_marked_lines': ssml_clean_lines,  # Keep markers as-is for proper highlighting
                'processed_lines': processed_lines
            }
        }

    def get_tts_lines(self, text):
        result = self.process_text(text)

        if 'line_by_line' not in result:
            return []

        line_data = result['line_by_line']
        tts_lines = []

        for i, (input_line, show_line, ssml_line) in enumerate(zip(
            line_data['input_lines'],
            line_data['show_text_lines'],
            line_data['ssml_clean_lines']
        )):
            if input_line.strip():  # Skip empty lines
                tts_lines.append({
                    'line_number': i + 1,
                    'input': input_line,
                    'show_text': show_line,
                    'tts_text': ssml_line,
                    'character_count': len(ssml_line),
                    'within_openai_limit': len(ssml_line) <= 4096,
                    'within_google_limit': len(ssml_line) <= 5000
                })

        return tts_lines

    def _track_actual_changes(self, original_words, final_words, changes_dict):
        import difflib
        import re

        matcher = difflib.SequenceMatcher(None, original_words, final_words)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                original_phrase = ' '.join(original_words[i1:i2])
                final_phrase = ' '.join(final_words[j1:j2])
                if original_phrase != final_phrase:
                    clean_original = original_phrase
                    clean_final = final_phrase

                    is_already_covered = False
                    for existing_key, existing_value in changes_dict.items():
                        if clean_original in existing_key and clean_final in existing_value:
                            is_already_covered = True
                            break

                    if not is_already_covered:
                        changes_dict[clean_original] = clean_final
            elif tag == 'delete':
                pass
            elif tag == 'insert':
                if i1 > 0 and i1 < len(original_words):
                    prev_word = original_words[i1-1] if i1 > 0 else ''
                    inserted_phrase = ' '.join(final_words[j1:j2])
                    if prev_word and len(inserted_phrase) > len(prev_word):
                        clean_prev = prev_word
                        clean_inserted = inserted_phrase

                        is_already_covered = False
                        for existing_key, existing_value in changes_dict.items():
                            if clean_prev in existing_key:
                                is_already_covered = True
                                break

                        if not is_already_covered:
                            changes_dict[clean_prev] = f"{clean_prev} {clean_inserted}"

    def _format_tts_for_debug(self, tts_text, max_line_length=50):
        if not tts_text:
            return ""

        input_lines = tts_text.split('\n')
        formatted_lines = []

        for line in input_lines:
            if not line.strip():
                formatted_lines.append("")  # Preserve empty lines
                continue

            words = line.split()
            current_line = ""
            line_parts = []

            for word in words:
                if current_line and len(current_line + " " + word) > max_line_length:
                    line_parts.append(current_line)
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word

            if current_line:
                line_parts.append(current_line)

            formatted_lines.extend(line_parts)

        return "\n".join(formatted_lines)

    def process_numbers_with_noun_context(self, text, track_changes=False, table=None):
        """
        New number-first approach: Find all numbers, then look for governing nouns
        to determine the appropriate conversion function.
        Starting with אחוז (percentage) as test case.
        """
        import re

        # Currency normalization is now handled by standardize_currency_terms() early in the pipeline
        # This reduces the complex regex patterns needed here

        number_changes = {}
        processed_text = text

        if not hasattr(self, '_processed_date_patterns'):
            self._processed_date_patterns = set()

        def is_match_inside_highlight_block(match, text):
            """Check if a regex match is inside an existing ~...~ highlight block"""
            start_pos = match.start()
            end_pos = match.end()

            highlight_blocks = []
            for block_match in re.finditer(r'~(.+?)~', text):
                highlight_blocks.append((block_match.start(), block_match.end()))

            for block_start, block_end in highlight_blocks:
                if block_start <= start_pos < block_end or block_start < end_pos <= block_end:
                    return True
            return False

        # Pattern for ל prefix to month name (e.g., "ה-8 לינואר") - consistent with ב pattern
        hebrew_date_pattern_prefix_to_month = r'([א-ת]+)-(\d{1,2})\s+ל([א-ת\u0591-\u05C7]+)(?:\s+(\d{4}))?'

        # Debug: Check if pattern matches
        matches = list(re.finditer(hebrew_date_pattern_prefix_to_month, processed_text))
        if matches:
            print(f"[DEBUG] HEBREW DATE PREFIX-TO-MONTH PATTERN found {len(matches)} matches in '{processed_text}'")

        for match in reversed(matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING match inside highlight block: {match.group(0)}")
                continue
            hebrew_prefix = match.group(1)  # ה
            day_str = match.group(2)        # 8
            month_name = match.group(3)     # ינואר (without ל)
            year_str = match.group(4)       # 2025 (optional)

            print(f"[DEBUG] PROCESSING HEBREW DATE ל-PATTERN: prefix='{hebrew_prefix}' day='{day_str}' month='{month_name}' year='{year_str}'")

            try:
                day_num = int(day_str)

                if 1 <= day_num <= 31:
                    # Convert day to Hebrew using existing method
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    if hebrew_prefix == "מה":
                        hebrew_prefix = "מְהַ"
                    elif hebrew_prefix == "מ":
                        hebrew_prefix = "מְהָ"
                    converted = f"{hebrew_prefix}{day_text} ל{month_name}"

                    if year_str:
                        try:
                            year_num = int(year_str)
                            if 1000 <= year_num <= 9999:
                                year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)  # Years are feminine
                                if hebrew_prefix == "מה":
                                    hebrew_prefix = "מְהַ"
                                elif hebrew_prefix == "מ":
                                    hebrew_prefix = "מְהָ"
                                converted = f"{hebrew_prefix}{day_text} ל{month_name} {year_text}"
                        except ValueError:
                            pass  # Keep without year if conversion fails

                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    self._processed_date_patterns.add(match.group(0))

                    print(f"[DEBUG] HEBREW DATE PREFIX-TO-MONTH REPLACEMENT: '{match.group(0)}' -> '{ssml_result}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), ssml_result, "HEBREW_DATE_PREFIX_MONTH")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        year_context_pattern = r'(בשנת|שנת|בשנה|שנה)\s+(\d{4})'

        for match in reversed(list(re.finditer(year_context_pattern, processed_text))):
            if is_match_inside_highlight_block(match, processed_text):
                continue
            context_word = match.group(1)  # בשנת
            year_str = match.group(2)      # 2025

            try:
                year_num = int(year_str)

                if 1900 <= year_num <= 2100:
                    year_text = self.heb2num(year_num, 'm', is_year=True)

                    converted = f"{context_word} {year_text}"

                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    self._processed_date_patterns.add(match.group(0))

                    print(f"[DEBUG] YEAR CONTEXT REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "YEAR_CONTEXT")

            except ValueError:
                continue  # Skip invalid years

        hebrew_date_pattern_ordinal = r'([א-ת]+)-(\d{1,2})\s+ב([א-ת\u0591-\u05C7]+)(?:\s+(\d{4}))?'

        for match in reversed(list(re.finditer(hebrew_date_pattern_ordinal, processed_text))):
            if is_match_inside_highlight_block(match, processed_text):
                continue
            hebrew_char = match.group(1)  # ה
            day_str = match.group(2)      # 6
            month_name = match.group(3)   # יולי
            year_str = match.group(4)     # 2025

            try:
                day_num = int(day_str)

                if 1 <= day_num <= 31:
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    if hebrew_char == "מה":
                        hebrew_char = "מְהַ"
                    elif hebrew_char == "מ":
                        hebrew_char = "מְהָ"

                    if year_str:
                        try:
                            year_num = int(year_str)
                            if 1900 <= year_num <= 2100:
                                year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                                converted = f"{hebrew_char}{day_text} ב{month_name} {year_text}"
                            else:
                                continue
                        except ValueError:
                            continue
                    else:
                        converted = f"{hebrew_char}{day_text} ב{month_name}"

                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] HEBREW DATE ORDINAL REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "HEBREW_DATE_ORDINAL")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Pattern for ה-DD.MM.YYYY format (e.g., "ה-10.04.2025") - Process FULL dates first
        date_patterns = self.get_consolidated_date_patterns()
        hebrew_ddmmyyyy_pattern = date_patterns['hebrew_prefix_dd_mm_yyyy']

        # Process Hebrew DD.MM.YYYY patterns FIRST (longer patterns have priority)
        ddmmyyyy_matches = list(re.finditer(hebrew_ddmmyyyy_pattern, processed_text))
        if ddmmyyyy_matches:
            print(f"[DEBUG] HEBREW DD.MM.YYYY PATTERN found {len(ddmmyyyy_matches)} matches in '{processed_text}'")

        for match in reversed(ddmmyyyy_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING DD.MM.YYYY match inside highlight block: {match.group(0)}")
                continue
            hebrew_prefix = match.group(1)  # ו-
            day_str = match.group(2)        # 15
            month_str = match.group(3)      # 03
            year_str = match.group(4)       # 2025

            print(f"[DEBUG] PROCESSING HEBREW DD.MM.YYYY PATTERN: prefix='{hebrew_prefix}' day='{day_str}' month='{month_str}' year='{year_str}'")

            try:
                converted = self._safe_date_processor().convert_date_pattern(day_str, month_str, year_str, hebrew_prefix.rstrip('-־'))
                if converted:
                    processed_text = processed_text.replace(match.group(0), converted)
                    print(f"[DEBUG] HEBREW DD.MM.YYYY REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "HEBREW_DD_MM_YYYY")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Process DD.MM.YYYY patterns BEFORE MM.YYYY patterns (longer patterns have priority)
        dd_mm_yyyy_pattern = date_patterns['dd_mm_yyyy']
        dd_mm_yyyy_matches = list(re.finditer(dd_mm_yyyy_pattern, processed_text))
        if dd_mm_yyyy_matches:
            print(f"[DEBUG] DD.MM.YYYY PATTERN found {len(dd_mm_yyyy_matches)} matches in '{processed_text}'")

        for match in reversed(dd_mm_yyyy_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING DD.MM.YYYY match inside highlight block: {match.group(0)}")
                continue

            day_str = match.group(1)    # 11
            month_str = match.group(2)  # 03
            year_str = match.group(3)   # 2025

            print(f"[DEBUG] PROCESSING DD.MM.YYYY PATTERN: day='{day_str}' month='{month_str}' year='{year_str}'")

            try:
                day_num = int(day_str)
                month_num = int(month_str)
                year_num = int(year_str)

                if 1 <= day_num <= 31 and 1 <= month_num <= 12 and 1900 <= year_num <= 2100:
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)  # Use ordinals for days
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                    converted = f"{day_text} ב{month_name} {year_text}"

                    processed_text = processed_text.replace(match.group(0), converted)
                    print(f"[DEBUG] DD.MM.YYYY REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "DD_MM_YYYY")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Process כמו DD.MM patterns first (more specific)
        kmo_dd_mm_pattern = date_patterns['kmo_dd_mm']
        kmo_dd_mm_matches = list(re.finditer(kmo_dd_mm_pattern, processed_text))
        if kmo_dd_mm_matches:
            print(f"[DEBUG] כמו DD.MM PATTERN found {len(kmo_dd_mm_matches)} matches in '{processed_text}'")

        for match in reversed(kmo_dd_mm_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING כמו DD.MM match inside highlight block: {match.group(0)}")
                continue

            context_word = match.group(1)  # כמו
            day_str = match.group(2)       # 5
            month_str = match.group(3)     # 4

            print(f"[DEBUG] PROCESSING כמו DD.MM PATTERN: context='{context_word}' day='{day_str}' month='{month_str}'")

            try:
                day_num = int(day_str)
                month_num = int(month_str)

                if 1 <= day_num <= 31 and 1 <= month_num <= 12:
                    dp = self._safe_date_processor()
                    day_text = dp.convert_day_to_hebrew(day_num)  # Use ordinals for days
                    month_name = dp.convert_month_to_hebrew(month_num)
                    converted = f"{context_word} {day_text} ל{month_name}"  # כמו חמישי לאפריל

                    processed_text = processed_text.replace(match.group(0), converted)
                    print(f"[DEBUG] כמו DD.MM REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "KMO_DD_MM")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Process YYYY.MM.DD patterns FIRST (before DD.MM to avoid conflicts)
        patterns = self.get_consolidated_date_patterns()
        yyyy_mm_dd_pattern = patterns['yyyy_mm_dd']

        for match in reversed(list(re.finditer(yyyy_mm_dd_pattern, processed_text))):
            year_str = match.group(1)
            month_str = match.group(2)
            day_str = match.group(3)

            try:
                year_num = int(year_str)
                month_num = int(month_str)
                day_num = int(day_str)

                if 1900 <= year_num <= 2100 and 1 <= month_num <= 12 and 1 <= day_num <= 31:
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                    converted = self._safe_date_processor().format_hebrew_date(day_text, month_name, year_text)

                    processed_text = processed_text[:match.start()] + converted + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] YYYY-MM-DD DATE REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    # Store directly in table if available
                    if table:
                        stored = self._store_conversion_in_table(table, match.group(0), converted, "YYYY-MM-DD")
                        if not stored and self.debug:
                            print(f"    ❌ Could not store '{match.group(0)}' in table")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Pattern for ה-DD.MM format (e.g., "ה-10.04" = day 10, month 04) - Process FIRST (more specific)
        hebrew_ddmm_pattern = date_patterns['hebrew_prefix_dd_mm']

        # Debug: Check if pattern matches
        ddmm_matches = list(re.finditer(hebrew_ddmm_pattern, processed_text))
        if ddmm_matches:
            print(f"[DEBUG] HEBREW DD.MM PATTERN found {len(ddmm_matches)} matches in '{processed_text}'")

        for match in reversed(ddmm_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING HEBREW DD.MM match inside highlight block: {match.group(0)}")
                continue

            # Check if this is actually a percentage (% symbol after the match)
            match_end = match.end()
            if match_end < len(processed_text) and processed_text[match_end] == '%':
                print(f"[DEBUG] SKIPPING HEBREW DD.MM match - detected percentage: {match.group(0)}%")
                continue

            hebrew_prefix = match.group(1)  # ה
            day_str = match.group(2)        # 10
            month_str = match.group(3)      # 04

            print(f"[DEBUG] PROCESSING HEBREW DD.MM PATTERN: prefix='{hebrew_prefix}' day='{day_str}' month='{month_str}'")

            try:
                day_num = int(day_str)
                month_num = int(month_str)

                if 1 <= day_num <= 31 and 1 <= month_num <= 12:
                    # Convert day to Hebrew using existing method
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    # Convert month number to Hebrew month name using existing method
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)

                    if hebrew_prefix == "מה":
                        hebrew_prefix = "מְהַ"
                    elif hebrew_prefix == "ה-":
                        # Special format for ה-DD/MM: ה-11/4 → האחד עשר לרביעי
                        converted = f"ה{day_text} ל{month_name}"
                        processed_text = processed_text[:match.start()] + converted + processed_text[match.end():]

                        if track_changes:
                            number_changes[match.group(0)] = converted

                        self._processed_date_patterns.add(match.group(0))
                        print(f"[DEBUG] HEBREW ה-DD.MM REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                        if table:
                            self._store_conversion_in_table(table, match.group(0), converted, "HEBREW_HEH_DD_MM")
                        continue

                    converted = f"{hebrew_prefix}{day_text} ב{month_name}"

                    processed_text = processed_text[:match.start()] + converted + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    self._processed_date_patterns.add(match.group(0))
                    print(f"[DEBUG] HEBREW DD.MM REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "HEBREW_DD_MM")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Process DD.MM patterns (day.month format like 5.4 = 5th of April) - Process AFTER Hebrew patterns
        dd_mm_pattern = date_patterns['dd_mm']
        dd_mm_matches = list(re.finditer(dd_mm_pattern, processed_text))
        if dd_mm_matches:
            print(f"[DEBUG] DD.MM PATTERN found {len(dd_mm_matches)} matches in '{processed_text}'")

        for match in reversed(dd_mm_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING DD.MM match inside highlight block: {match.group(0)}")
                continue

            # Skip if immediately followed by % (percentage like 9.5%)
            if match.end() < len(processed_text) and processed_text[match.end()] == '%':
                print(f"[DEBUG] SKIPPING DD.MM match – looks like percentage: {match.group(0)}%")
                continue

            day_str = match.group(1)    # 5
            month_str = match.group(2)  # 4

            print(f"[DEBUG] PROCESSING DD.MM PATTERN: day='{day_str}' month='{month_str}'")

            try:
                day_num = int(day_str)
                month_num = int(month_str)

                if 1 <= day_num <= 31 and 1 <= month_num <= 12:
                    dp = self._safe_date_processor()
                    day_text = dp.convert_day_to_hebrew(day_num)  # Use ordinals for days
                    month_name = dp.convert_month_to_hebrew(month_num)
                    converted = f"{day_text} ל{month_name}"  # Use ל prefix for month

                    processed_text = processed_text.replace(match.group(0), converted)
                    print(f"[DEBUG] DD.MM REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "DD_MM")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Process MM.YYYY patterns BEFORE general number processing
        mm_yyyy_pattern = date_patterns['mm_yyyy']
        mm_yyyy_matches = list(re.finditer(mm_yyyy_pattern, processed_text))
        if mm_yyyy_matches:
            print(f"[DEBUG] MM.YYYY PATTERN found {len(mm_yyyy_matches)} matches in '{processed_text}'")

        for match in reversed(mm_yyyy_matches):
            if is_match_inside_highlight_block(match, processed_text):
                print(f"[DEBUG] SKIPPING MM.YYYY match inside highlight block: {match.group(0)}")
                continue

            month_str = match.group(1)  # 12
            year_str = match.group(2)   # 2025

            print(f"[DEBUG] PROCESSING MM.YYYY PATTERN: month='{month_str}' year='{year_str}'")

            try:
                month_num = int(month_str)
                year_num = int(year_str)

                if 1 <= month_num <= 12 and 1900 <= year_num <= 2100:
                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)
                    converted = f"{month_name} {year_text}"

                    processed_text = processed_text.replace(match.group(0), converted)
                    print(f"[DEBUG] MM.YYYY REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    if table:
                        self._store_conversion_in_table(table, match.group(0), converted, "MM_YYYY")

            except (ValueError, KeyError):
                continue  # Skip invalid dates



        hebrew_date_pattern = r'ב-(\d{1,2})\s+ל([א-ת\u0591-\u05C7]+)\s+(\d{4})'

        for match in reversed(list(re.finditer(hebrew_date_pattern, processed_text))):
            if is_match_inside_highlight_block(match, processed_text):
                continue
            day_str = match.group(1)
            month_name = match.group(2)
            year_str = match.group(3)

            try:
                day_num = int(day_str)
                year_num = int(year_str)

                if 1 <= day_num <= 31 and 1900 <= year_num <= 2100:
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    converted = f"ב{day_text} ל{month_name} {year_text}"

                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] HEBREW DATE REPLACEMENT: '{match.group(0)}' -> '{ssml_result}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), ssml_result, "HEBREW_DATE")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        def handle_prefixed_patterns(text):
            """
            General prefix handler that:
            1. Finds any prefix followed by hyphen
            2. Removes prefix and processes the clean text with existing patterns
            3. Adds prefix back to the result
            """
            prefix_pattern = r'([^-\s]+)-(\d{1,2}\s+ב[א-ת\u0591-\u05C7]+\s+\d{4})'

            def process_prefixed_match(match):
                prefix = match.group(1)  # The prefix part
                content = match.group(2)  # The content after the hyphen (full date string)
                original_full = match.group(0)  # The full match

                date_pattern = r'(\d{1,2})\s+ב([א-ת\u0591-\u05C7]+)\s+(\d{4})'
                date_match = re.match(date_pattern, content)

                if date_match:
                    day_str = date_match.group(1)
                    month_name = date_match.group(2)
                    year_str = date_match.group(3)

                    try:
                        day_num = int(day_str)
                        year_num = int(year_str)

                        if 1 <= day_num <= 31 and 1900 <= year_num <= 2100:
                            # Convert day to Hebrew using ORDINAL for 1-10, cardinal for 11+
                            if 1 <= day_num <= 10:
                                day_text = self.text_processor.ORDINAL_MASCULINE[day_num]
                            else:
                                day_text = self.heb2num(day_num, 'm')

                            year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                            clean_converted = f"{prefix}{day_text} ב{month_name} {year_text}"
                            converted = clean_converted

                            tts_alias = clean_converted

                            if track_changes:
                                number_changes[original_full] = tts_alias

                            return converted

                    except (ValueError, KeyError):
                        pass

                return original_full

            return re.sub(prefix_pattern, process_prefixed_match, text)

        processed_text = handle_prefixed_patterns(processed_text)

        hebrew_date_pattern2 = r'(\d{1,2})\s+ב([א-ת\u0591-\u05C7]+)\s+(\d{4})'

        for match in reversed(list(re.finditer(hebrew_date_pattern2, processed_text))):
            match_start = match.start()
            match_end = match.end()

            day_str = match.group(1)
            month_name = match.group(2)
            year_str = match.group(3)

            try:
                day_num = int(day_str)
                year_num = int(year_str)

                if 1 <= day_num <= 31 and 1900 <= year_num <= 2100:
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    converted = self._safe_date_processor().format_hebrew_date(day_text, month_name, year_text)
                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] HEBREW DATE REPLACEMENT: '{match.group(0)}' -> '{ssml_result}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), ssml_result, "HEBREW_DATE_FORMAT")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        hebrew_month_pattern = r'חודש\s+([א-ת\u0591-\u05C7]+)\s+(\d{4})'

        for match in reversed(list(re.finditer(hebrew_month_pattern, processed_text))):
            month_name = match.group(1)
            year_str = match.group(2)

            try:
                year_num = int(year_str)

                if 1900 <= year_num <= 2100:
                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    converted = f"חודש {month_name} {year_text}"

                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] HEBREW MONTH REPLACEMENT: '{match.group(0)}' -> '{ssml_result}'")

                    # Store directly in table if available
                    if table:
                        self._store_conversion_in_table(table, match.group(0), ssml_result, "HEBREW_MONTH")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        # Date normalization is now handled early in pipeline by normalize_all_dates_to_dots()

        # Use centralized pattern from get_consolidated_date_patterns()
        patterns = self.get_consolidated_date_patterns()
        dd_mm_yyyy_pattern = patterns['dd_mm_yyyy']

        for match in reversed(list(re.finditer(dd_mm_yyyy_pattern, processed_text))):
            day_str = match.group(1)
            month_str = match.group(2)
            year_str = match.group(3)

            try:
                day_num = int(day_str)
                month_num = int(month_str)
                year_num = int(year_str)

                if self._safe_date_processor().validate_date_components(day_num, month_num, year_num):
                    day_text = self._safe_date_processor().convert_day_to_hebrew(day_num)

                    month_name = self._safe_date_processor().convert_month_to_hebrew(month_num)

                    year_text = self._safe_date_processor().convert_year_to_hebrew(year_num)

                    converted = self._safe_date_processor().format_hebrew_date(day_text, month_name, year_text)
                    ssml_result = converted

                    processed_text = processed_text[:match.start()] + ssml_result + processed_text[match.end():]

                    if track_changes:
                        number_changes[match.group(0)] = converted

                    print(f"[DEBUG] DD-MM-YYYY DATE REPLACEMENT: '{match.group(0)}' -> '{converted}'")

                    # Store directly in table if available
                    if table:
                        # Try to store with the match as-is (with hyphens)
                        stored = self._store_conversion_in_table(table, match.group(0), converted, "DD-MM-YYYY")

                        # If not found, try with slash version (original format before normalization)
                        if not stored:
                            slash_version = match.group(0).replace('-', '/')
                            if self.debug:
                                print(f"    [DEBUG] Trying slash version: '{slash_version}'")
                            self._store_conversion_in_table(table, slash_version, converted, "DD-MM-YYYY")

            except (ValueError, KeyError):
                continue  # Skip invalid dates

        currency_nouns = self._get_currency_nouns()
        currency_pattern = '|'.join(currency_nouns)
        decimal_currency_pattern = rf'([א-ת]+-?)?(\d+)\.(\d+)\s+({currency_pattern})'

        for match in reversed(list(re.finditer(decimal_currency_pattern, processed_text))):
            prefix = match.group(1) if match.group(1) else ''
            whole_part = match.group(2)
            decimal_part = match.group(3)
            currency_noun = match.group(4)

            currency_gender = self._hebrew_nouns_gender.get(currency_noun, 'm')  # Default to masculine if not found
            if currency_gender is None:
                currency_gender = 'm'  # Default to masculine if not found

            whole_text = self.heb2num(int(whole_part), currency_gender, construct_state=True)

            clean_prefix = prefix.rstrip('-') if prefix else ''

            is_israeli_currency = self._hebrew_nouns_types.get(currency_noun, '') == 'שקלים'
            if is_israeli_currency:
                decimal_normalized = decimal_part.ljust(2, '0') if len(decimal_part) == 1 else decimal_part
                decimal_value = int(decimal_normalized)

                # Special case for 1 shekel: use singular "שקל אחד" instead of "אחד שקלים"
                if whole_part == "1":
                    if decimal_value == 0:
                        hebrew_result = f"{clean_prefix}שקל אחד"
                    elif decimal_value == 1:
                        # 1.01 שקלים -> שקל אחד ואגורה אחת
                        hebrew_result = f"{clean_prefix}שקל אחד ואגורה אחת"
                    elif decimal_value == 2:
                        # 1.02 שקלים -> שקל אחד ושתי אגורות
                        hebrew_result = f"{clean_prefix}שקל אחד ושתי אגורות"
                    else:
                        decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                        hebrew_result = f"{clean_prefix}שקל אחד ו{decimal_text} אגורות"
                else:
                    if decimal_value == 0:
                        hebrew_result = f"{clean_prefix}{whole_text} שקלים"
                    elif decimal_value == 1:
                        # Special case for 1 agora: "אגורה אחת" instead of "אחת אגורות"
                        hebrew_result = f"{clean_prefix}{whole_text} שקלים ואגורה אחת"
                    elif decimal_value == 2:
                        # Special case for 2 agorot: "שתי אגורות" instead of "שתיים אגורות"
                        hebrew_result = f"{clean_prefix}{whole_text} שקלים ושתי אגורות"
                    else:
                        decimal_text = self.heb2num(decimal_value, 'f')  # feminine for אגורות
                        hebrew_result = f"{clean_prefix}{whole_text} שקלים ו{decimal_text} אגורות"
            else:
                decimal_value = int(decimal_part)
                if decimal_value == 0:
                    hebrew_result = f"{clean_prefix}{whole_text} {currency_noun}"
                else:
                    decimal_text = self.heb2num(decimal_value, currency_gender)
                    hebrew_result = f"{clean_prefix}{whole_text} נקודה {decimal_text} {currency_noun}"

            original = match.group(0)
            replacement = hebrew_result
            processed_text = processed_text[:match.start()] + replacement + processed_text[match.end():]

            if track_changes:
                number_changes[original] = replacement

            print(f"[DEBUG] DECIMAL CURRENCY EARLY REPLACEMENT: '{original}' -> '{replacement}'")

            # Store directly in table if available
            if table:
                self._store_conversion_in_table(table, original, replacement, "DECIMAL_CURRENCY_EARLY")
            print(f"[DEBUG] NUMBER: '{original}' | HEBREW: '{replacement}'")

        number_pattern = r'(\d+(?:\.\d+)?)'
        number_matches = list(re.finditer(number_pattern, processed_text))

        for match in reversed(number_matches):
            number_str = match.group(1)
            number_start = match.start()
            number_end = match.end()

            text_before_match = processed_text[:number_start]
            text_after_match = processed_text[number_end:]

            tildes_before = text_before_match.count('')
            tildes_after = text_after_match.count('')

            if tildes_before % 2 == 1:
                last_opening_tilde = text_before_match.rfind('')
                if last_opening_tilde != -1:
                    text_after_opening = text_before_match[last_opening_tilde+1:]
                    if '' not in text_after_opening:
                        print(f"[DEBUG] SKIPPING NUMBER '{number_str}' at {number_start}: inside ... markers")
                        continue

            skip_number = False
            if hasattr(self, '_processed_date_patterns'):
                for pattern in self._processed_date_patterns:
                    if number_str in pattern:
                        pattern_parts = pattern.split(number_str)
                        if len(pattern_parts) >= 2:
                            prefix_part = pattern_parts[0]  # e.g., "מה-"
                            text_snippet = processed_text[max(0, number_start-len(prefix_part)):number_end+5]
                            if prefix_part in text_snippet and text_snippet.find(prefix_part) + len(prefix_part) == text_snippet.find(number_str):
                                print(f"[DEBUG] SKIPPING NUMBER '{number_str}' at {number_start}: part of processed date pattern '{pattern}'")
                                skip_number = True
                                break

            if skip_number:
                continue

            # CRITICAL FIX: Check if this number is part of a dictionary phrase before converting
            # Look for potential 2-word phrases involving this number
            text_before_number = processed_text[:number_start].strip()
            if text_before_number:
                words_before = text_before_number.split()
                if words_before:
                    last_word = words_before[-1]
                    potential_phrase = f"{last_word} {number_str}"

                    # Check if this phrase exists in dictionary
                    phrase_result = self._search_raw_milon(potential_phrase)
                    if not phrase_result.empty:
                        print(f"[DEBUG] SKIPPING NUMBER '{number_str}' at {number_start}: part of dictionary phrase '{potential_phrase}'")
                        continue

            governing_info = self._find_governing_noun_with_position(processed_text, number_start, number_end)
            print(f"[DEBUG] FOUND NUMBER '{number_str}' at {number_start}-{number_end}, governing_info: {governing_info}")

            if governing_info:
                noun, noun_start, noun_end = governing_info
                noun_info = self._get_noun_type_and_gender(noun)
                if self.debug:
                    print(f"[DEBUG] Governing noun '{noun}' has info: {noun_info}")

                if noun_info and noun_info.get('type') == 'אחוז':
                    # TABLE-AWARE PROCESSING: Check if אחוז is already part of a compound in the table
                    compound_word = None
                    if table:
                        # Find the row containing אחוז
                        for i, row in enumerate(table.rows):
                            if row['source'] == noun and row.get('milon'):
                                milon_value = row['milon']
                                # Check if milon contains a compound phrase (multiple words)
                                if ' ' in milon_value and 'אחוזי' in milon_value:
                                    # Extract the second word from compound like "אחוזי נכות"
                                    parts = milon_value.split()
                                    if len(parts) >= 2 and parts[0] == 'אחוזי':
                                        compound_word = parts[1]
                                        if self.debug:
                                            print(f"[DEBUG] TABLE COMPOUND DETECTED: '{noun}' is part of '{milon_value}', extracted: '{compound_word}'")
                                        break

                    if compound_word:
                        # Compound percentage: "100 אחוז נכות" → "מאה אחוזי נכות"
                        number_hebrew = self.heb2num(int(number_str), 'm')  # Always masculine for percentages
                        result = f"{number_hebrew} אחוזי {compound_word}"

                        # Find the compound word in source text to include in replacement
                        compound_start = processed_text.find(compound_word, noun_end)
                        if compound_start != -1:
                            compound_end = compound_start + len(compound_word)
                            full_start = min(number_start, noun_start)
                            full_end = compound_end
                            original_text = processed_text[full_start:full_end]
                        else:
                            # Fallback if compound word not found adjacent
                            full_start = min(number_start, noun_start)
                            full_end = max(number_end, noun_end)
                            original_text = processed_text[full_start:full_end]

                        if self.debug:
                            print(f"[DEBUG] COMPOUND PERCENTAGE: '{original_text}' -> '{result}' (table compound: אחוזי {compound_word})")
                    else:
                        # Regular percentage like "100 אחוז" or "2.5%" → "מאה אחוזים" / "שניים וחצי אחוזים"
                        converted = self._convert_percentage_number(number_str, noun_info['gender'])
                        result = converted

                        full_start = min(number_start, noun_start)
                        full_end = max(number_end, noun_end)
                        original_text = processed_text[full_start:full_end]

                        if self.debug:
                            print(f"[DEBUG] REGULAR PERCENTAGE: '{original_text}' -> '{converted}' (no compound detected)")

                    if result:
                        tts_gender = "male"
                        processed_text = processed_text[:full_start] + result + processed_text[full_end:]

                        # Store directly in table if available
                        if table:
                            self._store_conversion_in_table(table, original_text, result, "PERCENTAGE")

                        if track_changes:
                            if noun == '%':
                                number_changes[f"{number_str}%"] = result
                            else:
                                number_changes[f"{number_str} {noun}"] = result

                elif noun_info and noun_info.get('type') == 'מספר שלם':
                    # Special handling for "1 שקלים" -> "שקל אחד"
                    if (number_str == "1" and noun in self._get_currency_nouns() and
                        self._hebrew_nouns_types.get(noun, '') == 'שקלים'):  # Handle Israeli currency
                        replacement_start = number_start
                        replacement_end = max(number_end, noun_end)  # Include the noun in replacement

                        # Handle Hebrew prefix (like ב-)
                        if (number_start >= 2 and
                            processed_text[number_start-1] == '-' and
                            '\u05D0' <= processed_text[number_start-2] <= '\u05EA'):  # Hebrew letter
                            replacement_start = number_start - 2  # Include the prefix
                            prefix = processed_text[number_start-2:number_start]
                            converted_with_prefix = f"{prefix.rstrip('-')}שקל אחד"
                        else:
                            converted_with_prefix = "שקל אחד"

                        result = converted_with_prefix
                        original_text = processed_text[replacement_start:replacement_end]
                        processed_text = processed_text[:replacement_start] + result + processed_text[replacement_end:]
                        print(f"[DEBUG] SINGULAR SHEKEL WHOLE NUMBER REPLACEMENT: '{original_text}' -> '{converted_with_prefix}' (1 שקלים -> שקל אחד)")

                        # Store directly in table if available
                        if table:
                            self._store_conversion_in_table(table, original_text, converted_with_prefix, "SINGULAR_SHEKEL_WHOLE")

                        if track_changes:
                            number_changes[original_text] = converted_with_prefix
                    else:
                        # Check if this is a date-related noun that should use ordinals
                        date_related_nouns = ['יום', 'תאריך']  # Add more date-related nouns as needed
                        governing_word = governing_info[0] if governing_info else None
                        if governing_word and governing_word in date_related_nouns:
                            # For ordinals, check immediate table context first, then fall back to governing word gender
                            immediate_context_gender = None
                            if table:
                                # Try to find immediate preceding context gender from the table
                                # We need to find the specific match for this number in the table
                                current_row_index = None

                                # Calculate what the full text would be (including prefixes)
                                if (number_start >= 2 and
                                    processed_text[number_start-1] == '-' and
                                    '\u05D0' <= processed_text[number_start-2] <= '\u05EA'):  # Hebrew letter
                                    # This is a prefixed number like "ה-10"
                                    search_text = processed_text[number_start-2:number_end]
                                else:
                                    # This is just the number
                                    search_text = number_str

                                # Find the exact match in the table
                                for idx, row in enumerate(table.rows):
                                    if row['source'] == search_text:
                                        current_row_index = idx
                                        if self.debug:
                                            print(f"    [MATCH] Found exact match '{search_text}' at table row {idx + 1}")
                                        break

                                # If no exact match, try to find by number only (fallback)
                                if current_row_index is None:
                                    for idx, row in enumerate(table.rows):
                                        if row['source'] == number_str:
                                            current_row_index = idx
                                            if self.debug:
                                                print(f"    [FALLBACK] Found number '{number_str}' at table row {idx + 1} (fallback)")
                                            break
                                if current_row_index is not None and current_row_index > 0:
                                    # Look backward for the most recent noun with gender (not just immediate preceding)
                                    for idx in range(current_row_index - 1, -1, -1):
                                        prev_row = table.rows[idx]
                                        prev_gender = prev_row.get('gender')
                                        if prev_gender and prev_gender not in ['None', '']:
                                            immediate_context_gender = prev_gender
                                            if self.debug:
                                                print(f"    [CONTEXT] Found preceding gender context: row {idx + 1} ('{prev_row['source']}') has gender '{prev_gender}' for number at row {current_row_index + 1}")
                                            break

                            # Use immediate context gender if available, otherwise use governing word gender
                            if immediate_context_gender:
                                gender_to_use = immediate_context_gender
                                gender_source = "immediate context"
                            else:
                                gender_to_use = 'm' if noun_info['gender'] == 'זכר' else 'f'
                                gender_source = f"governing word ({governing_word})"

                            converted = self._convert_date_ordinal_logic(int(number_str), gender_to_use)
                            if self.debug:
                                print(f"[DEBUG] _convert_date_ordinal_logic('{number_str}', '{gender_to_use}') returned: '{converted}' (date-related noun: {governing_word}, gender from: {gender_source})")
                        else:
                            # Check if this is a prefixed number (ב-3, ל-5, etc.)
                            is_prefixed_number = (number_start >= 2 and
                                                 processed_text[number_start-1] == '-' and
                                                 '\u05D0' <= processed_text[number_start-2] <= '\u05EA')  # Hebrew letter

                            if is_prefixed_number:
                                # For prefixed numbers, determine construct state based on prefix and noun
                                gender_code = 'm' if noun_info['gender'] == 'זכר' else 'f'
                                
                                # Get the prefix character
                                prefix_char = processed_text[number_start-2] if number_start >= 2 else ''
                                
                                # Use construct state for:
                                # 1. ש- prefix (relative pronoun "that")
                                # 2. Numbers with definite article ה
                                use_construct_for_prefix = (
                                    prefix_char == 'ש' or  # ש-2 שקלים = ששני שקלים 
                                    (noun and noun.startswith('ה'))  # ל-5 התשלומים
                                )
                                
                                converted = self.heb2num(int(number_str), gender_code, construct_state=use_construct_for_prefix)
                                if self.debug:
                                    print(f"[DEBUG] PREFIXED heb2num('{number_str}', '{gender_code}', construct_state={use_construct_for_prefix}) returned: '{converted}' (prefixed number, noun='{noun}')")
                            else:
                                converted = self._convert_whole_number(number_str, noun_info['gender'], noun)
                                if self.debug:
                                    print(f"[DEBUG] _convert_whole_number('{number_str}', '{noun_info['gender']}', '{noun}') returned: '{converted}'")
                        if converted:
                            tts_gender = "female" if noun_info['gender'] == 'נקבה' else "male"

                            result = converted

                            replacement_start = number_start
                            replacement_end = number_end

                            if (number_start >= 2 and
                                processed_text[number_start-1] == '-' and
                                '\u05D0' <= processed_text[number_start-2] <= '\u05EA'):  # Hebrew letter

                                if (replacement_end < len(processed_text) and
                                    processed_text[replacement_end] == ' '):
                                    replacement_end += 1  # Include the space in replacement
                                    print(f"[DEBUG] Prefix detected: including space in replacement for '{processed_text[number_start-2:number_end+1]}'")
                                else:
                                    print(f"[DEBUG] Prefix detected but no trailing space for '{processed_text[number_start-2:number_end]}'")
                            else:
                                print(f"[DEBUG] No Hebrew prefix with hyphen detected for number at {number_start}")

                            # Get original text before modification
                            original_text = processed_text[replacement_start:replacement_end]

                            # Store the complete conversion in table if available
                            if table:
                                # For prefixed numbers, find the full Hebrew prefix (supporting multi-character prefixes)
                                prefix_start = self._find_prefix_start(processed_text, number_start)
                                if prefix_start < number_start:
                                    # Include the full prefix in the table lookup
                                    table_key = processed_text[prefix_start:number_end]  # e.g., "ול-1000"
                                    table_value = processed_text[prefix_start:number_start] + result  # e.g., "ול-אֶלֶף"
                                else:
                                    table_key = original_text.strip()
                                    table_value = result

                                if self.debug:
                                    print(f"[DEBUG] PREFIXED_NUMBER REPLACEMENT: '{table_key}' -> '{table_value}'")
                                self._store_conversion_in_table(table, table_key, table_value, "PREFIXED_NUMBER")

                            processed_text = processed_text[:replacement_start] + result + processed_text[replacement_end:]
                            if track_changes:
                                number_changes[number_str] = converted

                elif noun_info and noun_info.get('type') == 'תאריך':
                    # Special rule: if noun ends with ם or ת, use same format as אגורות instead of date format
                    if noun.endswith('ם') or noun.endswith('ת'):
                        # Use exact same format as אגורות: construct state for 2, regular for others
                        gender_code = 'f' if noun_info['gender'] == 'נקבה' else 'm'
                        num_value = int(number_str)
                        if num_value == 2:
                            # Use construct state for 2 (like שתי אגורות)
                            converted = self.heb2num(num_value, gender_code, construct_state=True)
                        else:
                            # Use regular state for other numbers (like שלוש אגורות)
                            converted = self.heb2num(num_value, gender_code)
                        if converted:
                            tts_gender = "female" if noun_info['gender'] == 'נקבה' else "male"

                            result = converted

                            original_text = processed_text[number_start:number_end]
                            processed_text = processed_text[:number_start] + result + processed_text[number_end:]
                            print(f"[DEBUG] REGULAR FORMAT for '{noun}' (ends with ם/ת): '{original_text}' -> '{converted}' (gender: {noun_info.get('gender')}) - like אגורות")

                            # Store directly in table if available
                            if table:
                                self._store_conversion_in_table(table, original_text, converted, "REGULAR_LIKE_AGOROT")

                            if track_changes:
                                number_changes[number_str] = converted
                    else:
                        # Use regular date format (ordinals for 1-10, cardinals for 11+)
                        converted = self._convert_date_number(number_str, noun_info['gender'])
                        if converted:
                            tts_gender = "female" if noun_info['gender'] == 'נקבה' else "male"

                            result = converted

                            original_text = processed_text[number_start:number_end]
                            processed_text = processed_text[:number_start] + result + processed_text[number_end:]
                            print(f"[DEBUG] DATE REPLACEMENT: '{original_text}' -> '{converted}' (type: {noun_info.get('type')}, gender: {noun_info.get('gender')})")

                            # Store directly in table if available
                            if table:
                                self._store_conversion_in_table(table, original_text, converted, "DATE_NUMBER")

                            if track_changes:
                                number_changes[number_str] = converted

                elif noun_info and noun_info.get('type') == 'ספרות':
                    converted = self._convert_digits_number(number_str, noun_info['gender'])
                    if converted:
                        tts_gender = "female" if noun_info['gender'] == 'נקבה' else "male"

                        result = converted

                        original_text = processed_text[number_start:number_end]
                        processed_text = processed_text[:number_start] + result + processed_text[number_end:]
                        print(f"[DEBUG] DIGITS REPLACEMENT: '{original_text}' -> '{converted}' (type: {noun_info.get('type')}, gender: {noun_info.get('gender')})")

                        # Store directly in table if available
                        if table:
                            self._store_conversion_in_table(table, original_text, converted, "DIGITS_NUMBER")

                        if track_changes:
                            number_changes[number_str] = converted

                elif noun_info and 'gender' in noun_info:
                    if '.' in number_str and noun in self._get_currency_nouns():
                        print(f"[DEBUG] SKIPPING DECIMAL CURRENCY NUMBER '{number_str}' with noun '{noun}': should be handled by decimal currency processing")
                        continue

                    # Special handling for "1 שקלים" -> "שקל אחד"
                    if (number_str == "1" and noun in self._get_currency_nouns() and
                        self._hebrew_nouns_types.get(noun, '') == 'שקלים'):  # Handle Israeli currency
                        replacement_start = number_start
                        replacement_end = max(number_end, noun_end)  # Include the noun in replacement

                        # Handle Hebrew prefix (like ב-)
                        if (number_start >= 2 and
                            processed_text[number_start-1] == '-' and
                            '\u05D0' <= processed_text[number_start-2] <= '\u05EA'):  # Hebrew letter
                            replacement_start = number_start - 2  # Include the prefix
                            prefix = processed_text[number_start-2:number_start]
                            converted_with_prefix = f"{prefix.rstrip('-')}שקל אחד"
                        else:
                            converted_with_prefix = "שקל אחד"

                        result = converted_with_prefix
                        original_text = processed_text[replacement_start:replacement_end]
                        processed_text = processed_text[:replacement_start] + result + processed_text[replacement_end:]
                        print(f"[DEBUG] SINGULAR SHEKEL REPLACEMENT: '{original_text}' -> '{converted_with_prefix}' (1 שקלים -> שקל אחד)")

                        # Store directly in table if available
                        if table:
                            self._store_conversion_in_table(table, original_text, converted_with_prefix, "SINGULAR_SHEKEL")

                        if track_changes:
                            number_changes[original_text] = converted_with_prefix
                    else:
                        # Special rule: if noun ends with ם or ת, use construct state format (like אגורות)
                        use_construct_state = noun.endswith('ם') or noun.endswith('ת')
                        if use_construct_state:
                            # Convert number for construct state (like אגורות)
                            try:
                                if '.' in number_str:
                                    whole_num = int(float(number_str))
                                else:
                                    whole_num = int(number_str)
                                gender_code = 'm' if noun_info['gender'] == 'זכר' else 'f'
                                converted = self.heb2num(whole_num, gender_code, construct_state=True)
                                if self.debug:
                                    print(f"[DEBUG] Using construct state for '{noun}' (ends with ם/ת): '{number_str}' -> '{converted}'")
                            except:
                                converted = None
                        else:
                            converted = self._convert_regular_noun_number(number_str, noun_info['gender'])
                        if converted:
                            tts_gender = "female" if noun_info['gender'] == 'נקבה' else "male"

                            replacement_start = number_start
                            replacement_end = max(number_end, noun_end)  # Include the noun in replacement

                            # Handle Hebrew prefix (like ב-)
                            if (number_start >= 2 and
                                processed_text[number_start-1] == '-' and
                                '\u05D0' <= processed_text[number_start-2] <= '\u05EA'):  # Hebrew letter
                                replacement_start = number_start - 2  # Include the prefix
                                prefix = processed_text[number_start-2:number_start]
                                converted_with_prefix = f"{prefix.rstrip('-')}{converted}"
                            else:
                                converted_with_prefix = converted

                            # Create the complete result including the noun
                            result = f"{converted_with_prefix} {noun}"

                            original_text = processed_text[replacement_start:replacement_end]
                            processed_text = processed_text[:replacement_start] + result + processed_text[replacement_end:]
                            print(f"[DEBUG] REGULAR NOUN REPLACEMENT: '{original_text}' -> '{converted_with_prefix} {noun}' (type: {noun_info.get('type')}, gender: {noun_info.get('gender')}, noun: {noun})")

                            # Store directly in table if available
                            if table:
                                self._store_conversion_in_table(table, original_text, f"{converted_with_prefix} {noun}", "REGULAR_NOUN")

                            if track_changes:
                                number_changes[original_text] = f"{converted_with_prefix} {noun}"

        return processed_text, number_changes

    def _find_governing_noun_with_position(self, text, number_start, number_end):
        """
        Search 1-3 words back and forward from the number position
        to find governing nouns for different number types.
        Returns tuple: (noun, noun_start, noun_end) or None
        """
        import re

        number_str = text[number_start:number_end]

        if number_end < len(text) and text[number_end] == '%':
            return ('%', number_end, number_end + 1)

        if len(number_str) == 4 and number_str.isdigit():
            year_num = int(number_str)
            if 1900 <= year_num <= 2100:  # Reasonable year range
                month_names = list(self._hebrew_months.values())
                months_with_b = [f'ב{month}' for month in month_names]
                date_context_words = {'תאריך'} | set(months_with_b)

                context_start = max(0, number_start - 50)  # Look 50 characters back
                context_end = min(len(text), number_end + 50)  # Look 50 characters forward
                context_text = text[context_start:context_end]

                import re
                clean_context = context_text  # Input is already clean

                for month in month_names:
                    if (month in clean_context or
                        f'ב{month}' in clean_context or
                        f'ל{month}' in clean_context or
                        'תאריך' in clean_context):
                        return ('שנה', number_start, number_start)  # Virtual position

        words = re.findall(r'\S+', text)

        char_pos = 0
        number_word_index = -1
        for i, word in enumerate(words):
            word_start = char_pos
            word_end = char_pos + len(word)
            if word_start <= number_start < word_end:
                number_word_index = i
                break
            char_pos = word_end + 1  # +1 for space

        if number_word_index == -1:
            return None

        search_range = range(max(0, number_word_index - 3),
                           min(len(words), number_word_index + 4))

        candidates = []

        for i in search_range:
            if i == number_word_index:  # Skip the number word itself
                continue

            word = words[i].strip('.,!?;:')  # Remove punctuation

            import re
            clean_word = word  # Remove  markers

            word_without_he = clean_word
            if clean_word.startswith('ה') and len(clean_word) > 1:
                word_without_he = clean_word[1:]  # Remove the ה prefix

            matched_noun = None
            original_word_for_return = None
            if self._get_noun_type_and_gender(clean_word):
                matched_noun = clean_word
                original_word_for_return = clean_word  # Keep original with ה
            elif self._get_noun_type_and_gender(word_without_he):
                matched_noun = word_without_he
                original_word_for_return = clean_word  # Keep original with ה for definite article rule

            if matched_noun:
                word_char_pos = 0
                for j in range(i):
                    word_char_pos += len(words[j]) + 1  # +1 for space

                distance = abs(i - number_word_index)
                # Check if this noun has gender information (prioritize nouns with gender info)
                noun_info = self._get_noun_type_and_gender(matched_noun)
                has_gender_info = 1 if noun_info and noun_info.get('gender') else 0
                candidates.append((original_word_for_return, word_char_pos, word_char_pos + len(words[i]), distance, i, has_gender_info))

        if candidates:
            # Sort by: 1) Prefer nouns WITH gender info, 2) Distance, 3) Word position (later words first)
            candidates.sort(key=lambda x: (-x[5], x[3], -x[4]))  # -has_gender_info (1 first), distance, -word_index
            closest = candidates[0]
            return (closest[0], closest[1], closest[2])  # Return without distance and has_gender_info

        return None

    def _get_noun_type_and_gender(self, noun):
        """
        Get type and gender information for a noun from the Excel data.
        Returns a dictionary with 'type' and 'gender' keys, or None if not found.
        """
        if not noun:
            return None

        clean_noun = self._TextCleaningUtils.clean_noun_for_lookup(noun)

        noun_type = self._hebrew_nouns_types.get(clean_noun)
        noun_gender = self._hebrew_nouns_gender.get(clean_noun)

        if noun_type or noun_gender:
            result = {}
            if noun_type:
                if noun_type == 'שקלים':
                    result['type'] = 'מספר שלם'
                else:
                    result['type'] = noun_type
            if noun_gender:
                result['gender'] = 'זכר' if noun_gender == 'm' else 'נקבה'
            return result

        if noun == '%':
            return {'type': 'אחוז', 'gender': 'זכר'}

        if noun == 'שנה':
            return {'type': 'תאריך', 'gender': 'נקבה'}

        currency_nouns = {
            'שקלים': {'type': 'מספר שלם', 'gender': 'זכר'},
            # Israeli currency terms are now standardized early in pipeline
            'דולרים': {'type': 'מספר שלם', 'gender': 'זכר'},
            'דולר': {'type': 'מספר שלם', 'gender': 'זכר'},
            'יורו': {'type': 'מספר שלם', 'gender': 'זכר'},
            # Currency symbols are now standardized early in pipeline
            'אגורות': {'type': 'מספר שלם', 'gender': 'נקבה'},
            'אגורה': {'type': 'מספר שלם', 'gender': 'נקבה'}
        }

        if clean_noun in currency_nouns:
            return currency_nouns[clean_noun]

        return None

    def _convert_number_unified(self, number_str, conversion_type, gender):
        """
        Unified number conversion method that handles all number types:
        - percentage: "12%" → "שנים עשר אחוזים"
        - whole: "5" → "חמישה" (for nouns like תשלומים)
        - date: "5" → "חמישי" (1-10 use ordinals, 11+ use cardinals)
        - digits: "1234" → "אחד שתיים שלוש ארבע" (each digit separately)
        - regular: Basic number conversion based on noun gender
        """
        try:
            # Parse number (handle both whole and decimal)
            if '.' in number_str:
                whole_part, decimal_part = number_str.split('.')
                whole_num = int(whole_part)
                decimal_str = decimal_part
            else:
                whole_num = int(number_str)
                decimal_str = None

            gender_code = 'm' if gender == 'זכר' else 'f'

            # Handle different conversion types
            if conversion_type == 'percentage':
                # Reconstruct number string and use existing method
                if decimal_str:
                    number_str = f"{whole_num}.{decimal_str}"
                else:
                    number_str = str(whole_num)
                return self._convert_percentage_to_hebrew(number_str)

            elif conversion_type == 'date':
                return self._convert_date_ordinal_logic(whole_num, gender_code)

            elif conversion_type == 'digits':
                return self._convert_digits_logic(number_str, gender_code)

            elif conversion_type in ['whole', 'regular']:
                construct_state = (conversion_type == 'whole')
                hebrew_num = self.heb2num(whole_num, gender_code, construct_state=construct_state)
                return hebrew_num.strip() if hebrew_num else None

            else:
                # Default to regular conversion
                hebrew_num = self.heb2num(whole_num, gender_code)
                return hebrew_num.strip() if hebrew_num else None

        except:
            return None

    def _convert_date_ordinal_logic(self, whole_num, gender_code):
        """Date-specific ordinal conversion logic"""
        if 1 <= whole_num <= 10:
            # Use the centralized constants
            if gender_code == 'm':
                return HEBREW_ORDINAL_MASCULINE.get(whole_num)
            else:
                return HEBREW_ORDINAL_FEMININE.get(whole_num)
        else:
            hebrew_num = self.heb2num(whole_num, gender_code)
            return hebrew_num.strip() if hebrew_num else None

    def _convert_digits_logic(self, number_str, gender_code):
        """Digits-specific conversion logic (each digit separately)"""
        digit_maps = {
            'f': {'0': 'אפס', '1': 'אַחַת', '2': 'שתיים', '3': 'שלוש', '4': 'ארבע',
                  '5': 'חמש', '6': 'שש', '7': 'שבע', '8': 'שמונה', '9': 'תשע'},
            'm': {'0': 'אפס', '1': 'אחד', '2': 'שניים', '3': 'שלושה', '4': 'ארבעה',
                  '5': 'חמישה', '6': 'שישה', '7': 'שבעה', '8': 'שמונה', '9': 'תשעה'}
        }

        digit_map = digit_maps[gender_code]
        hebrew_digits = []

        for digit in number_str:
            if digit.isdigit():
                hebrew_digits.append(digit_map[digit])

        return ' '.join(hebrew_digits)

    # Legacy method wrappers for backward compatibility
    def _convert_percentage_number(self, number_str, gender):
        return self._convert_number_unified(number_str, 'percentage', gender)

    def _convert_date_number(self, number_str, gender):
        return self._convert_number_unified(number_str, 'date', gender)

    def _convert_whole_number_with_prefix(self, text, number_start, number_end, number_str, gender):
        """
        Convert a whole number to Hebrew with proper prefix handling.
        Handles cases like "ל-5" -> "לחמישה"
        """
        try:
            prefix = self._extract_prefix(text, number_start)

            num_value = int(number_str)
            # For prefixed numbers, construct state depends on the following noun having ה
            # This function doesn't have access to the following noun, so it uses absolute state by default
            # The correct construct state logic is handled in the main number processing functions
            hebrew_num = self.heb2num(num_value, gender, construct_state=False)
            hebrew_num = hebrew_num.strip() if hebrew_num else ""

            clean_prefix = prefix.rstrip('-') if prefix else ''

            return f"{clean_prefix}{hebrew_num}"
        except:
            return None

    def _find_prefix_start(self, text, number_start):
        """
        Find the start position of any Hebrew prefix before the number.
        Handles cases like "ל-5" and "ול-1000" where we need to include the full Hebrew prefix.
        """
        # Check if there's a hyphen right before the number
        if number_start >= 2 and text[number_start-1] == '-':
            # Look backwards to find the start of the Hebrew prefix
            prefix_start = number_start - 2  # Start before the hyphen
            while prefix_start >= 0 and '\u05D0' <= text[prefix_start] <= '\u05EA':  # Hebrew letter range
                prefix_start -= 1
            # prefix_start is now pointing to the character before the first Hebrew letter
            # so we add 1 to get the actual start of the Hebrew prefix
            return prefix_start + 1

        return number_start

    def _extract_prefix(self, text, number_start):
        """
        Extract any Hebrew prefix before the number.
        Returns the prefix string (like "ל-") or empty string if none.
        """
        prefix_start = self._find_prefix_start(text, number_start)
        if prefix_start < number_start:
            return text[prefix_start:number_start]
        return ""

    def _convert_whole_number(self, number_str, gender, noun=None):
        """
        Convert a number to Hebrew whole number format.
        For תשלומים and other whole number nouns.
        Uses construct state based on definite article rule:
        - If noun starts with ה (like התשלומים), use construct state (חֲמֵשֶׁת)
        - If no ה (like תשלומים), use absolute state (חֲמִישָׁה)
        """
        try:
            if '.' in number_str:
                whole_num = int(float(number_str))
            else:
                whole_num = int(number_str)

            gender_code = 'm' if gender == 'זכר' else 'f'

            # Apply definite article rule
            if noun:
                use_construct_state = noun.startswith('ה')
            else:
                # Fallback: default to construct state if no noun provided (backward compatibility)
                use_construct_state = True

            hebrew_num = self.heb2num(whole_num, gender_code, construct_state=use_construct_state)

            return hebrew_num.strip() if hebrew_num else None
        except:
            return None

    def _convert_digits_number(self, number_str, gender):
        """
        Convert a number to separate Hebrew digits.
        For ספרות: each digit is read individually (e.g., 7654 → שבע שש חמש ארבע)
        """
        try:
            gender_code = 'm' if gender == 'זכר' else 'f'

            digit_maps = {'f': {'0': 'אפס', '1': 'אַחַת', '2': 'שתיים', '3': 'שלוש', '4': 'ארבע', '5': 'חמש', '6': 'שש', '7': 'שבע', '8': 'שמונה', '9': 'תשע'}, 'm': {'0': 'אפס', '1': 'אחד', '2': 'שניים', '3': 'שלושה', '4': 'ארבעה', '5': 'חמישה', '6': 'שישה', '7': 'שבעה', '8': 'שמונה', '9': 'תשעה'}}

            digit_map = digit_maps[gender_code]

            hebrew_digits = []
            for digit in number_str:
                if digit.isdigit():
                    hebrew_digits.append(digit_map[digit])

            return ' '.join(hebrew_digits)

        except:
            return None

    def _convert_regular_noun_number(self, number_str, gender):
        """
        Convert a number to Hebrew for regular nouns.
        This is the basic number conversion based on noun gender.
        Used for nouns like שוברים, ספרים, etc. that have gender but no specific type.
        Uses construct state for numbers that directly govern nouns.
        """
        try:
            if '.' in number_str:
                whole_num = int(float(number_str))
            else:
                whole_num = int(number_str)

            gender_code = 'm' if gender == 'זכר' else 'f'

            hebrew_num = self.heb2num(whole_num, gender_code, construct_state=True)

            return hebrew_num.strip() if hebrew_num else None
        except:
            return None

    class _TextCleaningUtils:
        """Unified text cleaning utilities to consolidate similar methods"""

        @staticmethod
        def clean_hebrew_prefix(prefix):
            """Clean Hebrew prefix by removing hyphen and whitespace"""
            return prefix.rstrip('- \t') if prefix else ''

        @staticmethod
        def normalize_quotes(text):
            """Normalize quotes for consistent searching - handles apostrophes and Hebrew geresh"""
            return text.replace("'", "׳") if text else text

        @staticmethod
        def extract_word_parts(word):
            """Extract clean word with leading and trailing punctuation"""
            if not word:
                return word, "", ""

            if '<' in word:
                parts = word.split('<')
                if len(parts) == 2:
                    return parts[1], parts[0] + '<', ""

            word = re.sub(r'<[^>]+>', '', word)
            leading = ""
            i = 0
            while i < len(word) and (not word[i].isalnum() and word[i] != '_'):
                leading += word[i]
                i += 1
            trailing = ""
            j = len(word) - 1
            while j >= i and (not word[j].isalnum() and word[j] != '_'):
                trailing = word[j] + trailing
                j -= 1
            clean_word = word[i:j+1]
            clean_word = clean_word.strip("'")
            return clean_word, leading, trailing

        @staticmethod
        def clean_punctuation_basic(text):
            """Basic punctuation cleaning - handles double punctuation"""
            if not text:
                return text
            text = text.replace(',,', ',')
            text = text.replace('.,', '.')
            text = text.replace(',.', '.')
            text = re.sub(r'[,\.][,\.]', '.', text)  # Regex pattern
            text = text.replace(' , ', ', ')
            return text

        @staticmethod
        def clean_noun_for_lookup(noun):
            """Clean noun for dictionary lookup - removes markers"""
            if not noun:
                return noun

            clean_noun = noun  # Remove  markers
            return clean_noun.strip()

    def _get_currency_nouns(self):
        """
        Get currency nouns dynamically from the Excel data based on type information.
        Returns a list of nouns that are marked as currency-related.
        """
        currency_nouns = []

        for noun, noun_type in self._hebrew_nouns_types.items():
            if noun_type and 'שקלים' in noun_type:  # Currency type marked as "שקלים"
                currency_nouns.append(noun)

        fallback_currency_nouns = ['שקלים', 'דולרים', 'דולר', 'יורו', 'לירות', 'לירה', 'אגורות', 'אגורה']
        # Israeli currency variations are now standardized early in pipeline

        for noun in fallback_currency_nouns:
            if noun not in currency_nouns:
                currency_nouns.append(noun)

        currency_nouns.sort(key=len, reverse=True)

        return currency_nouns

    def _log_with_timing(self, message, line_num, start_time):
        """
        Extract timing logger from replace_with_punctuation.
        Logs processing steps with timing information for debugging.
        """
        import time

        current_time = time.perf_counter()
        if self.debug:  # Only print if debug mode is enabled
            if any(keyword in message.lower() for keyword in ['start', 'completed', 'processing time', 'word processing', 'punctuation', 'number', 'context', 'checking', 'lookup', 'match']):
                wall_time = time.time()
                timestamp = time.strftime("%H:%M:%S", time.localtime(wall_time))
                microseconds = int((wall_time % 1) * 1000000)
                full_timestamp = f"{timestamp}.{microseconds:06d}"

                # Calculate elapsed time since start and since last log using perf_counter for accuracy
                elapsed_since_start = (current_time - start_time) * 1000
                elapsed_since_last = (current_time - self._last_log_time) * 1000 if hasattr(self, '_last_log_time') else 0

                print(f"{full_timestamp} {line_num:3d}  {message:<60} (+{elapsed_since_last:6.3f}ms) [Total: {elapsed_since_start:7.3f}ms]")
        self._last_log_time = current_time
        return line_num + 1

__all__ = ["HebrewEnhanceTranslation", "aLobe"]
