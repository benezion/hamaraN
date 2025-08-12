import os
import re
import webbrowser
import html

# Helper: regex for Hebrew diacritics (nikud)
HEBREW_NIKUD = '\u0591-\u05C7'

def nikud_flexible_pattern(word):
    """
    Create a flexible regex pattern that matches Hebrew text with or without nikud.
    Handles complex phrases, multiple words, and various nikud combinations.
    """
    if not word:
        return ''

    # Handle HTML entities first
    word = word.replace('&#x27;', "'").replace('&#39;', "'")

    # If the word contains multiple words (spaces), handle each word separately
    if ' ' in word:
        words = word.split()
        patterns = []
        for w in words:
            if w.strip():  # Skip empty words
                patterns.append(nikud_flexible_pattern(w.strip()))
        # Join with flexible space pattern (allows for multiple spaces/nikud between words)
        return r'\s*'.join(patterns)

    # For single words, create flexible pattern
    pattern_chars = []

    for char in word:
        if char == ' ':
            # Space becomes flexible whitespace
            pattern_chars.append(r'\s+')
        elif char in 'אבגדהוזחטיכלמנסעפצקרשתךםןףץ':
            # Hebrew letter - allow optional nikud after it
            pattern_chars.append(f'{re.escape(char)}[֑-ׇ]*')
        elif char in '\'״"':
            # Apostrophe or quote - make it optional and flexible
            pattern_chars.append(r'(?:\'|&#x27;|&#39;|״|")?')
        elif char == 'ו' and len(word) > 1:
            # Special handling for vav - it might have different nikud
            pattern_chars.append(r'ו[֑-ׇ]*')
        elif char in '֑-ׇ':
            # Nikud character - make it optional
            pattern_chars.append(f'{re.escape(char)}?')
        else:
            # Other characters (numbers, punctuation, etc.)
            pattern_chars.append(re.escape(char))

    # Join all pattern parts
    pattern = ''.join(pattern_chars)

    # Make the entire pattern more flexible by allowing optional nikud at word boundaries
    # and handling potential HTML entities
    pattern = pattern.replace('(?:\'|&#x27;|&#39;|״|")?', r'(?:\'|&#x27;|&#39;|״|")?[֑-ׇ]*')

    return pattern

class DisplayProcessor:
    def __init__(self):
        pass

    def _strip_ssml_tags(self, text):
        """
        Remove SSML tags from text while preserving ~ markers for highlighting.
        This should be called before applying marker highlighting.
        """
        if not text:
            return text
        
        # Remove <say-as ...>...</say-as> tags but keep the inner content
        text = re.sub(r'<say-as[^>]*>(.*?)</say-as>', r'\1', text, flags=re.DOTALL)
        
        # Remove <speak>...</speak> tags but keep the inner content
        text = re.sub(r'<speak>(.*?)</speak>', r'\1', text, flags=re.DOTALL)
        
        # Remove any other remaining XML/SSML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        return text

    def text_to_html(self, text, highlight_dict=None, line_by_line=None, gender=None, original_input=None):
        """
        Convert text to HTML with highlighting for content between ~ markers.
        This is the new simple marker-based highlighting system.
        """
        if line_by_line:
            return self._text_to_html_line_by_line(text, highlight_dict, line_by_line, gender)
        else:
            return self._text_to_html_single_block(text, highlight_dict, gender, original_input)

    def _text_to_html_single_block(self, text, highlight_dict, gender=None, original_input=None):
        """Generate HTML with single-block display and marker-based highlighting"""
        
        def create_interleaved_lines(original_text, processed_text):
            """Create interleaved input/output lines"""
            if not original_text or not processed_text:
                return ""
            
            original_lines = original_text.split('\n')
            processed_lines = processed_text.split('\n')
            
            # Ensure both have the same number of lines
            max_lines = max(len(original_lines), len(processed_lines))
            while len(original_lines) < max_lines:
                original_lines.append("")
            while len(processed_lines) < max_lines:
                processed_lines.append("")
            
            interleaved_html = []
            for i, (orig_line, proc_line) in enumerate(zip(original_lines, processed_lines), 1):
                if not orig_line.strip() and not proc_line.strip():
                    continue  # Skip completely empty line pairs
                
                # Original line
                orig_content = orig_line if orig_line.strip() else "[Empty Line]"
                orig_class = "empty" if not orig_line.strip() else ""
                
                # Processed line  
                proc_content = proc_line if proc_line.strip() else "[Empty Line]"
                proc_class = "empty" if not proc_line.strip() else ""
                
                interleaved_html.append(f'''
                    <div class="line-pair">
                        <div class="numbered-line original-line {orig_class}">
                            <span class="line-number">{i}</span> 
                            <span class="line-label">מקור:</span>
                            <span class="line-content">{orig_content}</span>
                        </div>
                        <div class="numbered-line processed-line {proc_class}">
                            <span class="line-number">{i}</span> 
                            <span class="line-label">מעובד:</span>
                            <span class="line-content">{proc_content}</span>
                        </div>
                    </div>
                ''')
            
            return '\n'.join(interleaved_html)

        # Apply marker-based highlighting to processed text
        highlighted_text = self._apply_marker_highlighting(text)
        
        # Create interleaved content if original input is available
        content_section = ""
        if original_input:
            interleaved_content = create_interleaved_lines(original_input, highlighted_text)
            content_section = f"""
                <div class="section">
                    <h3>טקסט מקור ומעובד - שורה אחר שורה</h3>
                    <div class="interleaved-content">{interleaved_content}</div>
                </div>
            """
        else:
            # Fallback to old format if no original input
            def add_line_numbers(content):
                if not content:
                    return ""
                lines = content.split('\n')
                numbered_lines = []
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        numbered_lines.append(f'<div class="numbered-line"><span class="line-number">{i}</span> <span class="line-content">{line}</span></div>')
                    else:
                        numbered_lines.append(f'<div class="numbered-line empty"><span class="line-number">{i}</span> <span class="line-content">[Empty Line]</span></div>')
                return '\n'.join(numbered_lines)
            
            highlighted_text_numbered = add_line_numbers(highlighted_text)
            content_section = f"""
                <div class="section">
                    <h3>טקסט מעובד</h3>
                    <div class="text-content">{highlighted_text_numbered}</div>
                </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Hebrew Text Processing</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 10px; direction: rtl; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 8px; background-color: #fafafa; }}
                .section h3 {{ margin-top: 0; color: #333; text-align: center; }}
                .text-content {{ font-size: 1.4em; line-height: 1.2; padding: 10px; border-radius: 3px; }}
                .text-content.original {{ background-color: #e8f4f8; }}
                .text-content {{ background-color: #f9f9f9; }}
                .highlight {{ color: red; background-color: #ffeeee; padding: 2px 4px; border-radius: 3px; font-weight: bold; }}
                .line-pair {{ 
                    margin-bottom: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .numbered-line {{ 
                    display: flex; 
                    align-items: flex-start; 
                    padding: 6px 8px; 
                    min-height: 1.2em;
                }}
                .original-line {{ 
                    background-color: #fff59d; 
                    border-bottom: 1px solid #ccc;
                }}
                .processed-line {{ 
                    background-color: #ffffff; 
                }}
                .numbered-line.empty {{ 
                    opacity: 0.5; 
                }}
                .line-number {{ 
                    display: inline-block; 
                    min-width: 30px; 
                    text-align: right; 
                    color: #666; 
                    font-weight: bold; 
                    margin-left: 10px; 
                    padding: 2px 6px; 
                    background-color: #e8e8e8; 
                    border-radius: 3px; 
                    font-size: 0.8em; 
                    direction: ltr;
                    flex-shrink: 0;
                }}
                .line-label {{
                    display: inline-block;
                    min-width: 50px;
                    font-weight: bold;
                    color: #333;
                    margin-left: 10px;
                    flex-shrink: 0;
                }}
                .line-content {{
                    flex: 1;
                    padding-right: 10px;
                    word-wrap: break-word;
                }}
                .interleaved-content {{
                    font-size: 1.2em;
                    line-height: 1.2;
                }}
                .line-content {{
                    line-height: 1.3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>עיבוד טקסט עברי{f' - {gender}' if gender else ''}</h2>
                {content_section}
            </div>
        </body>
        </html>
        """
        return html

    def _text_to_html_line_by_line(self, text, highlight_dict, line_by_line, gender=None):
        """Generate HTML with line-by-line display using 2-line format (original + processed)"""
        # Get line-by-line data
        input_lines = line_by_line.get('input_lines', [])
        show_lines = line_by_line.get('show_text_lines', [])
        tts_lines = line_by_line.get('ssml_marked_lines', [])  # Use marked lines for highlighting

        html_lines = []

        for i, (input_line, show_line, tts_line) in enumerate(zip(input_lines, show_lines, tts_lines)):
            if not input_line.strip():  # Skip empty lines
                continue

            line_num = i + 1

            # Convert newlines to <br> tags for proper HTML display
            input_line_html = input_line.replace('\n', '<br>')
            tts_line_html = tts_line.replace('\n', '<br>')

            # Apply highlighting to the processed line
            highlighted_processed = self._apply_marker_highlighting(tts_line_html)

            # Use 2-line format: Original + Processed (with highlighting)
            html_lines.append(f"""
                <div class="line-pair">
                    <div class="numbered-line original-line">
                        <span class="line-number">{line_num}</span> 
                        <span class="line-label">מקור:</span>
                        <span class="line-content">{input_line_html}</span>
                    </div>
                    <div class="numbered-line processed-line">
                        <span class="line-number">{line_num}</span> 
                        <span class="line-label">מעובד:</span>
                        <span class="line-content">{highlighted_processed}</span>
                    </div>
                </div>
            """)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Hebrew Text Processing - Line by Line</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 10px; direction: rtl; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .line-pair {{ 
                    margin-bottom: 20px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .numbered-line {{ 
                    display: flex; 
                    align-items: flex-start; 
                    padding: 6px 8px; 
                    min-height: 1.2em;
                }}
                .original-line {{ 
                    background-color: #fff59d; 
                    border-bottom: 1px solid #ccc;
                }}
                .processed-line {{ 
                    background-color: #ffffff; 
                }}
                .numbered-line.empty {{ 
                    opacity: 0.5; 
                }}
                .line-number {{ 
                    display: inline-block; 
                    min-width: 30px; 
                    font-weight: bold; 
                    color: #666; 
                    margin-left: 8px; 
                    text-align: center;
                    direction: ltr;
                    flex-shrink: 0;
                }}
                .line-label {{
                    display: inline-block;
                    min-width: 50px;
                    font-weight: bold;
                    color: #333;
                    margin-left: 10px;
                    flex-shrink: 0;
                }}
                .line-content {{
                    flex: 1;
                    padding-right: 10px;
                    word-wrap: break-word;
                    line-height: 1.3;
                }}
                .highlight {{ color: red; background-color: #ffeeee; padding: 2px 4px; border-radius: 3px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>עיבוד טקסט עברי - שורה אחר שורה{f' - {gender}' if gender else ''}</h2>
                {''.join(html_lines)}
            </div>
        </body>
        </html>
        """
        return html

    def _apply_marker_highlighting(self, text):
        """
        Apply highlighting to text based on ~ markers, then remove ALL ~ characters.
        This is the core of the new simple highlighting system.
        """
        if not text:
            return text

        # FIRST: Strip SSML tags while preserving ~ markers
        text = self._strip_ssml_tags(text)

        # Find all content between ~ markers and highlight it
        def highlight_match(match):
            content = match.group(1)  # Content between ~ markers
            return f'<span class="highlight">{content}</span>'

        # Replace all ~content~ with highlighted content (this removes the ~ markers around highlighted content)
        highlighted_text = re.sub(r'~([^~]+)~', highlight_match, text)
        
        # Remove ALL remaining ~ characters (both paired and unpaired)
        highlighted_text = highlighted_text.replace('~', '')

        return highlighted_text

    def _apply_marker_highlighting_and_clean(self, text):
        """
        Apply highlighting to text based on ~ markers, then clean the markers from the text.
        This shows the highlighting but displays clean text (for TTS section).
        """
        if not text:
            return text

        # FIRST: Strip SSML tags while preserving ~ markers
        text = self._strip_ssml_tags(text)

        # Find all content between ~ markers and highlight it, but don't include the markers in the final text
        def highlight_match(match):
            content = match.group(1)  # Content between ~ markers
            return f'<span class="highlight">{content}</span>'

        # Handle both single and double tildes
        # Replace all ~~content~~ with highlighted content (removes ~~ markers)
        highlighted_text = re.sub(r'~~([^~]+)~~', highlight_match, text)
        # Replace all ~content~ with highlighted content (removes ~ markers)
        highlighted_text = re.sub(r'~([^~]+)~', highlight_match, highlighted_text)
        
        # Remove any remaining ~ characters
        highlighted_text = highlighted_text.replace('~', '')

        return highlighted_text

    def _apply_marker_highlighting_preserve_markers(self, text):
        """
        Apply highlighting to text based on ~ markers, but PRESERVE the ~ markers in the output.
        This is for the TTS section where you want to see both highlighting AND the markers.
        """
        if not text:
            return text

        # FIRST: Strip SSML tags while preserving ~ markers
        text = self._strip_ssml_tags(text)

        # Find all content between ~ markers and highlight it, but KEEP the markers visible
        def highlight_match(match):
            content = match.group(1)  # Content between ~ markers
            markers = match.group(0)  # Full match including ~ markers
            return f'<span class="highlight">{markers}</span>'  # Highlight the entire thing including ~

        # Replace all ~content~ with highlighted content that INCLUDES the ~ markers
        highlighted_text = re.sub(r'~([^~]+)~', highlight_match, text)
        
        # Don't remove any remaining ~ characters - keep them visible
        
        return highlighted_text

    def _clean_stray_markers_simple(self, text):
        """
        Simple approach to clean stray ~ markers while preserving complete pairs
        """
        if not text:
            return text

        # Count ~ characters
        tilde_count = text.count('~')
        if tilde_count % 2 == 0:
            # Even number - all pairs are complete
            return text

        # Odd number - remove the last stray ~
        # Find the last ~ that doesn't have a pair
        result = text
        while result.count('~') % 2 == 1:
            # Find the last ~
            last_tilde = result.rfind('~')
            if last_tilde != -1:
                # Check if this ~ has a matching pair before it
                before_text = result[:last_tilde]
                if before_text.count('~') % 2 == 0:
                    # This is a stray ~ at the end, remove it
                    result = result[:last_tilde] + result[last_tilde + 1:]
                else:
                    # This ~ has a pair, look for the first stray ~
                    first_tilde = result.find('~')
                    result = result[:first_tilde] + result[first_tilde + 1:]
            else:
                break

        return result

    def save_html(self, html_content, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        # Open the HTML file in the default browser
        webbrowser.open('file://' + os.path.abspath(output_path))


