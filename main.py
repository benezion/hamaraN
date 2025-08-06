import argparse
import os
from datetime import datetime
from engine import HebrewEnhanceTranslation
from tts import TTSProcessor
from display import DisplayProcessor

def main():
    parser = argparse.ArgumentParser(description='Gender-aware Hebrew text processor with TTS and display.')
    parser.add_argument('gender', type=str, choices=['f2m', 'm2f', 'm2m', 'f2f'], help='Gender conversion mode (m2f, f2m, etc.)')
    parser.add_argument('--text', '-t', help='Text to process (command line input)')
    parser.add_argument('--file', '-f', help='Input file containing Hebrew text to process')
    parser.add_argument('--tts', action='store_true', help='Enable TTS')
    parser.add_argument('--display', action='store_true', help='Enable HTML display output')
    parser.add_argument('--voice_id', type=str, default=None, help='Voice ID for TTS (optional)')
    parser.add_argument('--output_html', type=str, default='./temp_files/output.html', help='Output HTML file path')
    parser.add_argument('--tts_engine', type=str, default='google', choices=['google', 'gtts'], help='TTS engine to use: google or gtts')

    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    # Note: Table processing is now the only option - clean output with no ~ characters
    args = parser.parse_args()

    # Get input text from either command line or file
    input_text = ""
    process_line_by_line = False

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
            # Enable line-by-line processing if file contains newlines
            process_line_by_line = '\n' in input_text
            if args.debug:
                print(f"[OK] Text loaded from file: {args.file}")
                if process_line_by_line:
                    print(f"[OK] Multi-line file detected - enabling line-by-line processing")
        except FileNotFoundError:
            print(f"[ERROR] File '{args.file}' not found!")
            return 1
        except Exception as e:
            print(f"[ERROR] Error reading file: {e}")
            return 1
    elif args.text:
        input_text = args.text
        # Enable line-by-line processing if text contains newlines
        process_line_by_line = '\n' in input_text
        if args.debug:
            print(f"[OK] Text provided via command line")
            if process_line_by_line:
                print(f"[OK] Multi-line input detected - enabling line-by-line processing")
    else:
        # Interactive mode - ask user to input text
        print("Enter Hebrew text to process (press Enter twice to finish):")
        lines = []
        while True:
            try:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        input_text = '\n'.join(lines)

    if not input_text.strip():
        print("Error: No text provided. Use --text for command line input or --file for file input.")
        parser.print_help()
        return 1

    if args.debug:
        print(f"\nProcessing text: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
        print(f"Gender mode: {args.gender}")
        print(f"{'-'*50}")

    # Initialize the translation processor
    translation = HebrewEnhanceTranslation(gender=args.gender)
    translation.debug = args.debug  # Pass debug flag to translation logic

    # Enable debug file logging if debug mode is on
    if args.debug:
        translation.enable_debug_file("debug.txt")
        print("[OK] Debug output will be saved to debug.txt")

    # Process the text - either line by line or as a whole
    if process_line_by_line:
        # Process each line individually for better highlighting
        lines = input_text.split('\n')
        processed_lines = []
        all_dict_words = {}

        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                if args.debug:
                    print(f"\n--- Processing line {i+1}: {line} ---")

                # Pass line number for unique hamara table filenames
                line_number = f"_{i+1:02d}"  # Zero-padded (01, 02, 03, etc.)
                line_result = translation.process_text_with_table(text=line.strip(), line_number=line_number)
                processed_lines.append(line_result)

                # Merge dictionary words from all lines
                if line_result.get('dict_words'):
                    all_dict_words.update(line_result['dict_words'])

        # Combine results for display and TTS
        combined_ssml_clean = '\n'.join([r['ssml_clean'] for r in processed_lines])
        combined_ssml_marked = '\n'.join([r.get('ssml_marked', r['ssml_clean']) for r in processed_lines])
        combined_show_text = '\n'.join([r['show_text'] for r in processed_lines])

        result = {
            'original_input': input_text,
            'show_text': combined_show_text,
            'ssml_clean': combined_ssml_clean,
            'ssml_marked': combined_ssml_marked,
            'ssml_text': combined_ssml_clean,
            'dict_words': all_dict_words,
            'processed_lines': processed_lines  # Keep individual line results for display
        }
    else:
        # Use NEW table-based processing instead of legacy system
        # Single line gets _01 suffix
        result = translation.process_text_with_table(text=input_text, line_number="_01")

    # SSML is already displayed in process_text_with_table method

    # Display
    if args.display:
        display = DisplayProcessor()

        # Determine which text to send to display
        display_text = None
        if process_line_by_line and 'processed_lines' in result:
            display_text = result['ssml_marked']
            # Create line-by-line display for file processing
            line_by_line_data = {
                'input_lines': [line.strip() for line in input_text.split('\n') if line.strip()],
                'show_text_lines': [r['show_text'] for r in result['processed_lines']],
                'ssml_marked_lines': [r.get('ssml_marked', r['ssml_clean']) for r in result['processed_lines']],
                'ssml_clean_lines': [r['ssml_clean'] for r in result['processed_lines']]
            }

            html_content = display.text_to_html(
                text=display_text,
                highlight_dict=result['dict_words'],
                line_by_line=line_by_line_data,
                gender=args.gender
            )
        elif 'line_by_line' in result:
            display_text = result['ssml_marked']
            # Original line-by-line processing
            html_content = display.text_to_html(
                text=display_text,  # Text with ~ markers for highlighting (will be cleaned in HTML)
                highlight_dict=result['dict_words'],
                line_by_line=result['line_by_line'],
                gender=args.gender
            )
        else:
            display_text = result.get('ssml_marked', result['ssml_clean'])
            # Fallback to simple single-block display
            html_content = display.text_to_html(
                text=display_text,  # Use marked version if available
                highlight_dict=result['dict_words'],
                gender=args.gender,
                original_input=result.get('original_input', input_text)
            )

        # Debug: Show buffer sent to display.py
        if args.debug:
            print(f"\n{'='*60}")
            print("📤 BUFFER SENT TO DISPLAY.PY:")
            print(f"'{display_text}'")
            print(f"\n{'='*60}")

        display.save_html(html_content, args.output_html)
        if args.debug:
            print(f"📄 HTML output saved to: {args.output_html}")

    # TTS
    if args.tts:
        tts_processor = TTSProcessor(engine=args.tts_engine, debug=args.debug)
        voice_id = args.voice_id if args.voice_id != "None" else None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sentence-based TTS generation with concatenation (default behavior)
        if args.debug:
            print(f"\n🎵 SENTENCE-BASED TTS MODE")
            print(f"{'='*60}")
        
        # Use local temp_files directory for sentence MP3s
        sentence_output_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(sentence_output_dir, exist_ok=True)
        
        sentence_result = tts_processor.text_to_speech_by_sentences(
            result['ssml_clean'],
            gender=args.gender[-1],
            voice_id=voice_id,
            output_dir=sentence_output_dir,
            base_name=f"sentence_{timestamp}",
            play_individual=False  # Don't play individual sentences, only the final combined file
        )
        
        if sentence_result:
            # Copy combined file to timestamped output
            output_mp3_path = f"./output_audio_{timestamp}.mp3"
            import shutil
            shutil.copy2(sentence_result['combined_file'], output_mp3_path)
            
            if args.debug:
                print(f"\n📊 SENTENCE TTS RESULTS:")
                print(f"  📝 Total sentences: {len(sentence_result['sentence_texts'])}")
                print(f"  🎵 Individual MP3s: {len(sentence_result['individual_files'])}")
                print(f"  ⏱️  Processing time: {sentence_result['total_time_ms']:.1f}ms")
                print(f"  📁 Individual files saved in: {os.path.dirname(sentence_result['individual_files'][0])}")
                print(f"  🎯 Combined file: {sentence_result['combined_file']}")
                print(f"  📋 Final output: {output_mp3_path}")
                
                # Show sentence breakdown
                print(f"\n📋 SENTENCE BREAKDOWN:")
                for i, (sentence, mp3_file) in enumerate(zip(sentence_result['sentence_texts'], sentence_result['individual_files'])):
                    print(f"  {i+1:2d}. '{sentence[:50]}...' → {os.path.basename(mp3_file)}")
            
            # Play the final combined file
            if args.debug:
                print(f"\n🎵 PLAYING FINAL COMBINED FILE...")
            tts_processor.play_audio(sentence_result['combined_file'])
            
            print(f"🎵 You can find your combined audio file at: {output_mp3_path}")
            if args.debug:
                print(f"🔊 Individual sentence files available in: {os.path.dirname(sentence_result['individual_files'][0])}")
        else:
            print(f"❌ Failed to generate sentence-based TTS")

    # Print results
    if args.debug:
        print(f"\n{'='*50}")
        print("RESULTS:")
        print(f"{'='*50}")
        print(f"Original text: {result['original_input']}")
        print(f"Processed text: {result['show_text']}")
        if result['dict_words']:
            print(f"Dictionary replacements: {len(result['dict_words'])} words")
            for orig, repl in result['dict_words'].items():
                print(f"  '{orig}' → '{repl}'")

    # Cleanup debug file
    if args.debug:
        # Note: disable_debug_file() was removed - debug file cleanup handled automatically
        print("[OK] Debug output saved to debug.txt")

    return result['ssml_text']

if __name__ == '__main__':
    result = main()
