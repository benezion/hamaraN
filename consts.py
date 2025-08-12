# -*- coding: utf-8 -*-
"""
Shared Hebrew constants and settings extracted from translationLobe.py
This lightweight module lets the rest of the codebase import constants without
parsing the giant translationLobe source.
"""

# === Hebrew constants ===
HEBREW_CONSTANTS = {
    'UNITS': {
        'm': ['', 'אֶחָד', 'שְׁנַיִם', 'שְׁלוֹשָׁה', 'אַרְבָּעָה', 'חֲמִישָּׁה', 'שִׁשָּׁה', 'שִׁבְעָה', 'שְׁמוֹנָה', 'תִּשְׁעָה'],
        'f': ['', 'אַחַת', 'שְׁתַּיִם', 'שָׁלוֹשׁ', 'אַרְבַּע', 'חָמֵשׁ', 'שֵׁשׁ', 'שֶׁבַע', 'שְׁמוֹנֶה', 'תֵּשַׁע']
    },
    'TEENS': {
        'm': ['עֲשָׂרָה', 'אַחַד עָשָׂר', 'שְׁנֵים עָשָׂר', 'שְׁלוֹשָׁה עָשָׂר', 'אַרְבָּעָה עָשָׂר',
              'חֲמִישָּׁה עָשָׂר', 'שִׁשָּׁה עָשָׂר', 'שִׁבְעָה עָשָׂר', 'שְׁמוֹנָה עָשָׂר', 'תִּשְׁעָה עָשָׂר'],
        'f': ['עֶשֶׂר', 'אַחַת עֶשְׂרֵה', 'שְׁתֵּים עֶשְׂרֵה', 'שְׁלוֹשׁ עֶשְׂרֵה', 'אַרְבַּע עֶשְׂרֵה',
              'חֲמֵשׁ עֶשְׂרֵה', 'שֵׁשׁ עֶשְׂרֵה', 'שְׁבַע עֶשְׂרֵה', 'שְׁמוֹנֶה עֶשְׂרֵה', 'תְּשַׁע עֶשְׂרֵה']
    },
    'TENS': ['', '', 'עֶשְׂרִים', 'שְׁלוֹשִׁים', 'אַרְבָּעִים', 'חֲמִישִׁים', 'שִׁשִּׁים', 'שִׁבְעִים', 'שְׁמוֹנִים', 'תִּשְׁעִים'],
    'HUNDREDS': ['', 'מֵאָה', 'מָאתַיִם', 'שְׁלוֹשׁ מֵאוֹת', 'אַרְבַּע מֵאוֹת', 'חֲמֵשׁ מֵאוֹת', 'שֵׁשׁ מֵאוֹת', 'שְׁבַע מֵאוֹת', 'שְׁמוֹנֶה מֵאוֹת', 'תְּשַׁע מֵאוֹת'],
    'CONSTRUCT_UNITS': {3: 'שְׁלוֹשֶׁת', 4: 'אַרְבַּעַת', 5: 'חֲמֵשֶׁת', 6: 'שֵׁשֶׁת', 7: 'שִׁבְעַת', 8: 'שְׁמוֹנַת', 9: 'תִּשְׁעַת', 10: 'עֲשֶׂרֶת'},
    'CONSTRUCT_UNITS_LOW': {
        'm': {1: 'אַחַד', 2: 'שְׁנֵי'},
        'f': {1: 'אַחַת', 2: 'שְׁתֵי'}
    },
    'ORDINAL_MASCULINE': {1: 'ראשון', 2: 'שֵנִי', 3: 'שלישִּי', 4: 'רביעי', 5: 'חמישִּי', 6: 'שישִּי', 7: 'שביעי', 8: 'שמינִּי', 9: 'תשיעי', 10: 'עשירי'},
    'ORDINAL_FEMININE': {1: 'ראשונה', 2: 'שנייה', 3: 'שלישית', 4: 'רביעית', 5: 'חמישית', 6: 'שישית', 7: 'שביעית', 8: 'שמינית', 9: 'תשיעית', 10: 'עשירית'},
    'MONTHS': {1: 'יָנוּאָר', 2: 'פֶבְּרוּאָר', 3: 'מֵרְץ', 4: 'אַפְּרִיל', 5: 'מַאי', 6: 'יוּנִי', 7: 'יוּלִי', 8: 'אוֹגוּסְט', 9: 'סֶפְּטֶמְבֶּר', 10: 'אוֹקְטוֹבֶּר', 11: 'נוֹבֶמְבֶּר', 12: 'דֵּצֶמְבֶּר'},
    'PREFIXES': ['ב', 'ל', 'מ', 'כ', 'ש', 'ה', 'ו', 'מה', 'וה', 'בה', 'לה', 'כה', 'שה'],
    'THOUSAND': 'אֶלֶף',
    'THOUSANDS': 'אֲלָפִים'
}

# Aliases for backwards compatibility
HEBREW_UNITS = HEBREW_CONSTANTS['UNITS']
HEBREW_TEENS = HEBREW_CONSTANTS['TEENS']
HEBREW_TENS = HEBREW_CONSTANTS['TENS']
HEBREW_ORDINAL_MASCULINE = HEBREW_CONSTANTS['ORDINAL_MASCULINE']
HEBREW_ORDINAL_FEMININE = HEBREW_CONSTANTS['ORDINAL_FEMININE']
HEBREW_HUNDREDS = HEBREW_CONSTANTS['HUNDREDS']
HEBREW_CONSTRUCT_UNITS = HEBREW_CONSTANTS['CONSTRUCT_UNITS']
HEBREW_CONSTRUCT_UNITS_LOW = HEBREW_CONSTANTS['CONSTRUCT_UNITS_LOW']
HEBREW_MONTHS = HEBREW_CONSTANTS['MONTHS']
HEBREW_PREFIXES = HEBREW_CONSTANTS['PREFIXES']
HEBREW_THOUSAND = HEBREW_CONSTANTS['THOUSAND']
HEBREW_THOUSANDS = HEBREW_CONSTANTS['THOUSANDS']

EXCEL_FILENAME = 'milon_zachar_nekeva_new.xlsx'
CURRENCY_FORMAT_TEMPLATE = "{prefix}{whole_text} שקלים ו{decimal_text} אגורות"

PERSON_INDICATORS = {
    'first': {'אני', 'שאני', 'ואני'},
    'second': {
        'male': {'אתה', 'שאתה', 'ואתה', 'כשאתה'},
        'female': {'אַתְּ', 'שאַתְּ', 'ואַתְּ', 'כשאַתְּ'},
        'neutral': {'שלך', 'איתך', 'לך', 'אותך', 'את/ה'}
    }
}

DEFAULT_NOUN_GENDERS = {
    'כוכבית': 'f', 'דקות': 'f', 'PSI': 'm', '%': 'm', 'אחוז': 'm', 'אחוזים': 'm',
    'שנה': 'f', 'שנים': 'f', 'חודש': 'm', 'חודשים': 'm', 'יום': 'm', 'ימים': 'm',
    'שעה': 'f', 'שעות': 'f', 'דקה': 'f', 'שניה': 'f', 'שניות': 'f'
}
