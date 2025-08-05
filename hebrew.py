
import pandas as pd
from consts import *
from table import TextProcessingTable
from dates import ConsolidatedDateProcessor

class HebrewTextProcessor:
    ORDINAL_MASCULINE = {
        1: 'ראשון', 2: 'שֵנִי', 3: 'שלישִּי', 4: 'רביעי', 5: 'חמישִּי',
        6: 'שישִּי', 7: 'שביעי', 8: 'שמינִּי', 9: 'תשיעי', 10: 'עֲשִׂירִיּ'
    }

    def __init__(self, gender=None, debug=False):
        self.global_context = {
            'detected_gender': None,
            'second_person_detected': False,
            'detected_in_word': None,
            'sentence_start_idx': 0,
            'force_maintain_gender': False
        }
        self.last_milon_gender: str | None = None
        self.gender = gender
        self.debug = debug

    def determine_column_and_gender(self, person_context, is_first_person, source_gender, target_gender, gender_context=None, is_verb_from_dictionary=False, is_neutral_second_person=False, person_value=None, debug_flag=False):
        if self.last_milon_gender in ["male", "female"]:
            if debug_flag:
                print(f"[DEBUG] last_milon_gender found: {self.last_milon_gender}")
            if debug_flag:
                print(f"[DEBUG] [last_milon_gender] Using and resetting last_milon_gender: {self.last_milon_gender}")
            column_to_use = 'Zachar' if self.last_milon_gender == 'male' else 'Nekeva'
            tts_gender = self.last_milon_gender
            if debug_flag:
                print(f"[DEBUG] *** last_milon_gender RESET to None (was: {tts_gender})")
            self.last_milon_gender = None
            if debug_flag:
                print(f"[DEBUG] Using column: {column_to_use}, tts_gender: {tts_gender}")
            return column_to_use, tts_gender
        if is_verb_from_dictionary and person_value is not None:
            if person_value == "ראשון":
                selected_gender = source_gender
                column_to_use = 'Zachar' if selected_gender == 'm' else 'Nekeva'
                tts_gender = "male" if selected_gender == 'm' else "female"
                return column_to_use, tts_gender
            elif person_value == "שני":
                selected_gender = target_gender
                column_to_use = 'Zachar' if selected_gender == 'm' else 'Nekeva'
                tts_gender = "male" if selected_gender == 'm' else "female"
                return column_to_use, tts_gender
        if gender_context is not None:
            column_to_use = 'Zachar' if gender_context == 'm' else 'Nekeva'
            tts_gender = "male" if gender_context == 'm' else "female"
            return column_to_use, tts_gender
        if is_neutral_second_person:
            column_to_use = 'Zachar' if target_gender == 'm' else 'Nekeva'
            tts_gender = "male" if target_gender == 'm' else "female"
            return column_to_use, tts_gender
        if person_context == "FIRST_PERSON" or is_first_person:
            column_to_use = 'Zachar' if source_gender == 'm' else 'Nekeva'
            tts_gender = "male" if source_gender == 'm' else "female"
        elif person_context == "SECOND_PERSON":
            column_to_use = 'Zachar' if target_gender == 'm' else 'Nekeva'
            tts_gender = "male" if target_gender == 'm' else "female"
        else:
            if source_gender != target_gender:
                column_to_use = 'Zachar' if target_gender == 'm' else 'Nekeva'
                tts_gender = "male" if target_gender == 'm' else "female"
            else:
                column_to_use = 'Zachar' if source_gender == 'm' else 'Nekeva'
                tts_gender = "male" if source_gender == 'm' else "female"
        return column_to_use, tts_gender

    def determine_inherent_gender(self, word, text=None, word_position=None, update_global=True):
        # Simplified: milon-based approach handles את disambiguation automatically
        if word in PERSON_INDICATORS['second']['male']:
            if update_global:
                self.global_context['detected_gender'] = 'male'
                self.global_context['second_person_detected'] = True
                self.global_context['detected_in_word'] = word
                self.global_context['force_maintain_gender'] = True
            return 'male'
        stripped_word = word.strip()
        for female_indicator in PERSON_INDICATORS['second']['female']:
            female_indicator = female_indicator.strip()
            if stripped_word == female_indicator:
                if update_global:
                    self.global_context['detected_gender'] = 'female'
                    self.global_context['second_person_detected'] = True
                    self.global_context['detected_in_word'] = word
                    self.global_context['force_maintain_gender'] = True
                return 'female'
        return None

    def should_maintain_original_gender(self, word, conversion_mode):
        if self.global_context['force_maintain_gender']:
            return True
        if self.global_context['second_person_detected']:
            # Check if detected gender matches the target gender (3rd character of conversion_mode)
            target_gender = conversion_mode[2] if len(conversion_mode) >= 3 else conversion_mode[0]
            detected_gender_letter = 'm' if self.global_context['detected_gender'] == 'male' else 'f'
            if target_gender == detected_gender_letter:
                return True
        return False

    def check_person_indicators(self, word, current_person_context, text=None, word_position=None):
        is_first_person = word in PERSON_INDICATORS['first']
        is_second_person = False
        inherent_gender = None
        is_verb_form = False
        is_neutral_second_person = False

        if self.debug:
            print(f"[DEBUG] check_person_indicators: word='{word}', is_first_person={is_first_person}")

        if word in PERSON_INDICATORS['second']['male']:
            is_second_person = True
            inherent_gender = 'male'
        elif word in PERSON_INDICATORS['second']['female']:
            is_second_person = True
            inherent_gender = 'female'
            if self.debug:
                print(f"[DEBUG] Matched female second person indicator: '{word}'")
        elif word in PERSON_INDICATORS['second']['neutral']:
            is_second_person = True
            is_neutral_second_person = True
            # For neutral indicators, use target gender from conversion mode
            target_gender = self.gender[2] if self.gender and len(self.gender) >= 3 else (self.gender[0] if self.gender and len(self.gender) >= 1 else 'f')
            inherent_gender = 'male' if target_gender == 'm' else 'female'
            if self.debug:
                print(f"[DEBUG] Matched neutral second person indicator: '{word}' -> using target gender: {inherent_gender}")

        if not is_second_person:
            if word.endswith('י') and len(word) >= 3:
                feminine_verb_patterns = ['תוכל', 'תרצ', 'תדבר', 'תלכ', 'תקח', 'תבוא', 'תצא',
                                         'תסביר', 'תאמר', 'תשאל', 'תענ', 'תלמד', 'תעש']
                for pattern in feminine_verb_patterns:
                    if word.startswith(pattern):
                        is_second_person = True
                        is_verb_form = True
                        inherent_gender = 'female'
                        break
            elif word.startswith('ת') and len(word) >= 3:
                masculine_verb_patterns = ['תוכל', 'תרצה', 'תדבר', 'תלך', 'תקח', 'תבוא', 'תצא',
                                          'תסביר', 'תאמר', 'תשאל', 'תענה', 'תלמד', 'תעשה']
                if any(word == pattern for pattern in masculine_verb_patterns):
                    is_second_person = True
                    is_verb_form = True
                    inherent_gender = 'male'
        person_context = current_person_context
        if is_first_person:
            person_context = "FIRST_PERSON"
            # Set global context for persistent gender across multiple words
            # In first person, speaker gender = first character of conversion mode
            source_gender = self.gender[0] if self.gender and len(self.gender) >= 1 else 'f'
            speaker_gender = 'female' if source_gender == 'f' else 'male'

            self.global_context['detected_gender'] = speaker_gender
            self.global_context['second_person_detected'] = False  # This is first person
            self.global_context['detected_in_word'] = word
            self.global_context['force_maintain_gender'] = True
            if self.debug:
                print(f"[DEBUG] *** Global context SET to '{speaker_gender}' by first person indicator: {word}")
        elif is_second_person:
            person_context = "SECOND_PERSON"
            # Set global context for persistent gender across multiple words
            # In second person, listener gender = third character of conversion mode
            target_gender = self.gender[2] if self.gender and len(self.gender) >= 3 else (self.gender[0] if self.gender and len(self.gender) >= 1 else 'f')
            listener_gender = 'female' if target_gender == 'f' else 'male'

            self.global_context['detected_gender'] = listener_gender
            self.global_context['second_person_detected'] = True
            self.global_context['detected_in_word'] = word
            self.global_context['force_maintain_gender'] = True
            if self.debug:
                print(f"[DEBUG] *** Global context SET to '{listener_gender}' by second person indicator: {word}")
        return person_context, is_first_person, is_second_person, inherent_gender, is_verb_form, is_neutral_second_person

    # REMOVED: _is_standalone_pronoun function
    # No longer needed - milon-based approach handles את disambiguation

    def process_nikud_replacement(self, row, person_context, is_first_person_indicator, source_gender, target_gender, source_tts_gender, target_tts_gender, gender_context, debug_flag=False):
        conversion_mode = f"{source_gender}2{target_gender}"
        if debug_flag:
            print(f"[DEBUG] process_nikud_replacement called with gender_context = {gender_context}, person_context = {person_context}")
            print(f"[DEBUG] source_gender = {source_gender}, target_gender = {target_gender}")
            print(f"[DEBUG] Global context: {self.global_context}")
        original_word = row.iloc[0]['Original'] if 'Original' in row.columns else None
        inherent_gender = None
        is_neutral_second_person = False
        if original_word and original_word in PERSON_INDICATORS['second']['neutral']:
            is_neutral_second_person = True
        is_verb_from_dictionary = False
        person_value = None

        # CRITICAL FIX: Only apply gender parameter conversion when person indicators are detected
        # Gender parameters (f2f, m2m, f2m, m2f) should ONLY activate when there are person indicators
        has_person_indicator = (
            self.global_context.get('detected_gender') is not None or  # Any person indicator detected
            self.global_context.get('second_person_detected') or        # Second person detected
            person_context in ["FIRST_PERSON", "SECOND_PERSON"] or     # Explicit person context
            is_first_person_indicator                                   # First person indicator
        )

        if debug_flag and source_gender != target_gender:
            print(f"[DEBUG] Gender conversion requested: {source_gender} -> {target_gender}")
            print(f"[DEBUG] Person indicator detected: {has_person_indicator}")
            if not has_person_indicator:
                print(f"[DEBUG] SKIPPING gender conversion - no person indicators detected")

        # PRIMARY GENDER CONVERSION LOGIC - Apply gender conversion ONLY if person indicators exist
        conversion_mode = f"{source_gender}2{target_gender}"
        if source_gender != target_gender and has_person_indicator:
            if debug_flag:
                print(f"[DEBUG] Gender conversion needed: {source_gender} -> {target_gender}")
            if source_gender == 'f' and target_gender == 'm':
                # Female to male conversion - use masculine form
                if 'Zachar' in row.columns and pd.notna(row.iloc[0]['Zachar']) and str(row.iloc[0]['Zachar']).strip():
                    replacement = row.iloc[0]['Zachar']
                    tts_gender = "male"
                    if debug_flag:
                        print(f"[DEBUG] {conversion_mode} conversion using Zachar column: '{replacement}' (tts_gender: {tts_gender})")
                        print(f"[DEBUG] DICT CONVERSION: {conversion_mode} - Using ZACHAR form: '{replacement}'")
                    return replacement, tts_gender
                elif 'זכר' in row.columns and pd.notna(row.iloc[0]['זכר']) and str(row.iloc[0]['זכר']).strip():
                    replacement = row.iloc[0]['זכר']
                    tts_gender = "male"
                    if debug_flag:
                        print(f"[DEBUG] {conversion_mode} conversion using זכר column: '{replacement}' (tts_gender: {tts_gender})")
                        print(f"[DEBUG] DICT CONVERSION: {conversion_mode} - Using זכר column: '{replacement}'")
                    return replacement, tts_gender
            elif source_gender == 'm' and target_gender == 'f':
                # Male to female conversion - use feminine form
                if 'Nekeva' in row.columns and pd.notna(row.iloc[0]['Nekeva']) and str(row.iloc[0]['Nekeva']).strip():
                    replacement = row.iloc[0]['Nekeva']
                    tts_gender = "female"
                    if debug_flag:
                        print(f"[DEBUG] {conversion_mode} conversion using Nekeva column: '{replacement}' (tts_gender: {tts_gender})")
                        print(f"[DEBUG] DICT CONVERSION: {conversion_mode} - Using NEKEVA form: '{replacement}'")
                    return replacement, tts_gender
                elif 'נקבה' in row.columns and pd.notna(row.iloc[0]['נקבה']) and str(row.iloc[0]['נקבה']).strip():
                    replacement = row.iloc[0]['נקבה']
                    tts_gender = "female"
                    if debug_flag:
                        print(f"[DEBUG] {conversion_mode} conversion using נקבה column: '{replacement}' (tts_gender: {tts_gender})")
                        print(f"[DEBUG] DICT CONVERSION: {conversion_mode} - Using נקבה column: '{replacement}'")
                    return replacement, tts_gender

        # Use gender columns if gender context is detected (אני/אתה/etc.)
        if self.global_context['second_person_detected']:
            detected_gender = self.global_context['detected_gender']
            if detected_gender == 'male':
                if 'Zachar' in row.columns and pd.notna(row.iloc[0]['Zachar']) and row.iloc[0]['Zachar'].strip():
                    replacement = row.iloc[0]['Zachar']
                    tts_gender = "male"
                    if debug_flag:
                        print(f"[DEBUG] DICT CONVERSION: detected male pronoun - Using ZACHAR form: '{replacement}'")
                    return replacement, tts_gender
                elif 'זכר' in row.columns and pd.notna(row.iloc[0]['זכר']) and str(row.iloc[0]['זכר']).strip():
                    replacement = row.iloc[0]['זכר']
                    tts_gender = "male"
                    if debug_flag:
                        print(f"[DEBUG] DICT CONVERSION: detected male pronoun - Using זכר column: '{replacement}'")
                    return replacement, tts_gender
            elif detected_gender == 'female':
                if 'Nekeva' in row.columns and pd.notna(row.iloc[0]['Nekeva']) and row.iloc[0]['Nekeva'].strip():
                    replacement = row.iloc[0]['Nekeva']
                    tts_gender = "female"
                    if debug_flag:
                        print(f"[DEBUG] DICT CONVERSION: detected female pronoun - Using NEKEVA form: '{replacement}'")
                    return replacement, tts_gender
                elif 'נקבה' in row.columns and pd.notna(row.iloc[0]['נקבה']) and str(row.iloc[0]['נקבה']).strip():
                    replacement = row.iloc[0]['נקבה']
                    tts_gender = "female"
                    if debug_flag:
                        print(f"[DEBUG] DICT CONVERSION: detected female pronoun - Using נקבה column: '{replacement}'")
                    return replacement, tts_gender

        if 'Person' in row.columns and pd.notna(row.iloc[0]['Person']):
            person_value_raw = row.iloc[0]['Person'].strip()
            if person_value_raw and person_value_raw.strip():
                person_value = person_value_raw
                if person_value == 'שני':
                    is_verb_from_dictionary = True
        if original_word and not is_verb_from_dictionary:
            inherent_gender = self.determine_inherent_gender(original_word)
            if inherent_gender:
                gender_context = inherent_gender
        maintain_original_gender = False
        if original_word and not is_verb_from_dictionary:
            maintain_original_gender = self.should_maintain_original_gender(original_word, conversion_mode)
        if 'Nikud' in row.columns and pd.notna(row.iloc[0]['Nikud']) and row.iloc[0]['Nikud'].strip() != '':
            replacement = row.iloc[0]['Nikud']

            if person_context == "SECOND_PERSON":
                if is_verb_from_dictionary:
                    if source_gender == 'f' and target_gender == 'm':
                        if 'Zachar' in row.columns and pd.notna(row.iloc[0]['Zachar']):
                            replacement = row.iloc[0]['Zachar']
                            if debug_flag:
                                print(f"[DEBUG] DICT LOOKUP: {conversion_mode} verb - overriding Nikud with Zachar value: '{replacement}'")
                        tts_gender = "male"
                    elif source_gender == 'm' and target_gender == 'f':
                        if 'Nekeva' in row.columns and pd.notna(row.iloc[0]['Nekeva']):
                            replacement = row.iloc[0]['Nekeva']
                            if debug_flag:
                                print(f"[DEBUG] DICT LOOKUP: {conversion_mode} verb - overriding Nikud with Nekeva value: '{replacement}'")
                        tts_gender = "female"
                    else:
                        tts_gender = target_tts_gender
                elif maintain_original_gender:
                    if inherent_gender == 'male':
                        tts_gender = "male"
                    elif inherent_gender == 'female':
                        tts_gender = "female"
                    else:
                        if self.global_context['detected_gender']:
                            gender_context = self.global_context['detected_gender']
                            tts_gender = "male" if gender_context == 'male' else "female"
                        else:
                            tts_gender = target_tts_gender
                elif inherent_gender == 'male' and source_gender == 'f' and target_gender == 'm':
                    tts_gender = "male"
                elif inherent_gender == 'female' and source_gender == 'm' and target_gender == 'f':
                    tts_gender = "female"
                elif source_gender == 'f' and target_gender == 'm':
                    tts_gender = "male"
                elif source_gender == 'm' and target_gender == 'f':
                    tts_gender = "female"
                else:
                    tts_gender = target_tts_gender
            else:
                # Use target gender when gender conversion has occurred
                if source_gender != target_gender:
                    tts_gender = target_tts_gender
                else:
                    tts_gender = source_tts_gender
            if debug_flag:
                print(f"[DEBUG] Using value: {replacement}")
            return replacement, tts_gender
        else:
            if is_verb_from_dictionary:
                if person_context == "SECOND_PERSON":
                    if source_gender == 'f' and target_gender == 'm':
                        column_to_use = 'Zachar'
                        tts_gender = "male"
                        if debug_flag:
                            print(f"[DEBUG] DICT LOOKUP: {conversion_mode} verb - using Zachar column")
                    elif source_gender == 'm' and target_gender == 'f':
                        column_to_use = 'Nekeva'
                        tts_gender = "female"
                        if debug_flag:
                            print(f"[DEBUG] DICT LOOKUP: {conversion_mode} verb - using Nekeva column")
                    else:
                        column_to_use = 'Zachar' if target_gender == 'm' else 'Nekeva'
                        tts_gender = "male" if target_gender == 'm' else "female"
                else:
                    column_to_use = 'Zachar' if source_gender == 'm' else 'Nekeva'
                    tts_gender = "male" if source_gender == 'm' else "female"
            elif inherent_gender:
                gender_context = inherent_gender
            elif maintain_original_gender and person_context == "SECOND_PERSON":
                if inherent_gender == 'male':
                    column_to_use = 'Zachar'
                    tts_gender = "male"
                elif inherent_gender == 'female':
                    column_to_use = 'Nekeva'
                    tts_gender = "female"
                elif self.global_context['detected_gender'] == 'male':
                    column_to_use = 'Zachar'
                    tts_gender = "male"
                elif self.global_context['detected_gender'] == 'female':
                    column_to_use = 'Nekeva'
                    tts_gender = "female"
                else:
                    column_to_use, tts_gender = self.determine_column_and_gender(
                        person_context, is_first_person_indicator, source_gender, target_gender,
                        gender_context, is_verb_from_dictionary, is_neutral_second_person, person_value, debug_flag
                    )
            else:
                column_to_use, tts_gender = self.determine_column_and_gender(
                    person_context, is_first_person_indicator, source_gender, target_gender,
                    gender_context, is_verb_from_dictionary, is_neutral_second_person, person_value, debug_flag
                )
            replacement = row.iloc[0][column_to_use]

            if pd.isna(replacement) or not str(replacement).strip():
                # Check if no gender context detected - use Nikud column if available
                if (not self.global_context['second_person_detected'] and
                    'Nikud' in row.columns and
                    pd.notna(row.iloc[0]['Nikud']) and
                    str(row.iloc[0]['Nikud']).strip()):
                    replacement = row.iloc[0]['Nikud']
                    if debug_flag:
                        print(f"[DEBUG] Column '{column_to_use}' is empty, no gender context detected, using Nikud: '{replacement}'")
                else:
                    replacement = row.iloc[0]['Original']
                    if debug_flag:
                        print(f"[DEBUG] Column '{column_to_use}' is empty, using original word: '{replacement}'")

            if debug_flag:
                print(f"[DEBUG] MILON replacement for phrase '{original_word}': '{replacement}' (TTS gender: {tts_gender})")
            return replacement, tts_gender



    # REMOVED: is_direct_object_marker function
    # No longer needed - milon-based approach with prefix stripping handles את disambiguation



    def reset_global_context(self, debug_flag=False):
        self.global_context = {
            'detected_gender': None,
            'second_person_detected': False,
            'detected_in_word': None,
            'sentence_start_idx': 0,
            'force_maintain_gender': False
        }
        self.last_milon_gender: str | None = None


__all__ = ["HebrewTextProcessor"]
