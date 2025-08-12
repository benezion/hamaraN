import argparse
import os
import time
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
    parser.add_argument('--tts_engine', type=str, default='google', choices=['google', 'gtts'], help='TTS engine to use: google (default) or gtts (gtts is faster)')
    parser.add_argument('--speed', type=float, default=1.0, help='TTS playback speed rate (1.0=normal, 1.1=10%% faster, 0.9=10%% slower)')
    
    # Additional gTTS quality parameters
    parser.add_argument('--slow', action='store_true', help='Generate slower, clearer speech (gTTS only)')
    parser.add_argument('--lang_check', action='store_true', help='Enable language validation (slower startup, gTTS only)')
    parser.add_argument('--tld', type=str, default='co.il', choices=['com', 'co.il', 'co.uk'], help='Google domain for accent: co.il=Israeli (default), com=standard, co.uk=British (gTTS only)')
    parser.add_argument('--sentence_tts', action='store_true', help='Process TTS sentence by sentence for better accuracy (slower but more accurate)')

    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--excel', action='store_true', help='Export processing table to Excel (hamara_table.xlsx)')
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

        # Keep results separate - no line combining
        result = {
            'original_input': input_text,
            'show_text': input_text,  # Show original input without combining
            'ssml_clean': input_text,  # Use original input
            'ssml_marked': input_text,  # Use original input  
            'ssml_text': input_text,  # Use original input
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


        display.save_html(html_content, args.output_html)
        if args.debug:
            print(f"📄 HTML output saved to: {args.output_html}")

    # TTS - ULTRA FAST MP3 GENERATION ⚡⚡⚡
    if args.tts:
        tts_processor = TTSProcessor(
            engine=args.tts_engine, 
            debug=args.debug, 
            speed_rate=args.speed,
            slow=args.slow,
            lang_check=args.lang_check,
            tld=args.tld
        )
        voice_id = args.voice_id if args.voice_id != "None" else None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.debug:
            print(f"\n🚀 LINE-BY-LINE TTS MODE (Always)")
            print(f"{'='*60}")
        
        # Check if input has multiple lines - always process line by line
        if process_line_by_line and 'processed_lines' in result:
            # For multi-line files: Use individually processed lines
            processed_clean_lines = [r['ssml_clean'] for r in result['processed_lines']]
            lines = [line.strip() for line in processed_clean_lines if line.strip()]
            text_to_process = '\n'.join(lines)  # Rejoin the processed lines
        else:
            # For single line or combined processing
            text_to_process = result['ssml_clean']
            lines = [line.strip() for line in text_to_process.split('\n') if line.strip()]
        
        if len(lines) > 1:
            # MULTI-LINE: Generate separate MP3 for each line and play sequentially
            if args.debug:
                print(f"🔄 Processing {len(lines)} lines individually with sequential playback...")
                if process_line_by_line:
                    print(f"🔧 Using individually processed lines (not split combined text)")
            
            line_result = tts_processor.text_to_speech_line_by_line(
                text_to_process,
                gender=args.gender[-1],
                voice_id=voice_id,
                output_dir=f"./line_audio_{timestamp}",
                base_name=f"line_{timestamp}",
                play_sequential=True  # Play each line one after another
            )
            
            if line_result and line_result['individual_files']:
                if args.debug:
                    print(f"✅ Generated {len(line_result['individual_files'])} individual line MP3 files")
                    print(f"✅ Sequential playback completed")
                    print(f"📁 Line files saved in: ./line_audio_{timestamp}/")
                print(f"🎵 Processed {len(line_result['individual_files'])} lines sequentially")
            else:
                print(f"❌ Failed to generate line-by-line TTS audio")
        else:
            # SINGLE LINE: Use standard TTS processing with cache optimization
            output_mp3_path = f"./output_audio_{timestamp}.mp3"
            start_time = time.time()
            
            # OPTIMIZATION 1: Check cache first (can save 2+ seconds!)
            cache_path = tts_processor._get_cache_path(result['ssml_clean'], args.tts_engine)
            if os.path.exists(cache_path):
                # Copy cached file to output location
                import shutil
                shutil.copy2(cache_path, output_mp3_path)
                processing_time = (time.time() - start_time) * 1000
                
                # Get file timestamps
                file_stats = os.stat(output_mp3_path)
                creation_time = datetime.fromtimestamp(file_stats.st_ctime)
                access_time = datetime.fromtimestamp(file_stats.st_atime)
                
                # Get MP3 duration
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(output_mp3_path)
                    duration_seconds = len(audio) / 1000.0
                    duration_formatted = f"{int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}"
                except Exception as e:
                    duration_formatted = "Unknown"
                    if args.debug:
                        print(f"[DEBUG] Could not get audio duration: {e}")
                
                if args.debug:
                    print(f"🚀 CACHE HIT! Used cached audio (saved ~2000ms)")
                    print(f"  📝 Text length: {len(result['ssml_clean'])} characters")
                    print(f"  ⚡ Processing time: {processing_time:.1f}ms (cached)")
                    print(f"  🎯 Output file: {output_mp3_path}")
                    print(f"  💾 File size: {os.path.getsize(output_mp3_path) / 1024:.1f} KB")
                    print(f"  🕐 Created: {creation_time.strftime('%H:%M:%S %d/%m/%Y')}")
                    print(f"  🕐 Accessed: {access_time.strftime('%H:%M:%S %d/%m/%Y')}")
                    print(f"  🎵 Duration: {duration_formatted}")
                # OPTIMIZATION 2: Async audio playback (non-blocking)
                tts_processor.play_audio_async(output_mp3_path)
                print(f"🎵 Audio file created at {creation_time.strftime('%H:%M:%S')} ({duration_formatted}) - Available at: {output_mp3_path}")
            else:
                # Generate new audio with optimizations
                if args.sentence_tts:
                    # SENTENCE-BY-SENTENCE TTS for better accuracy
                    if args.debug:
                        print(f"🔄 Using sentence-by-sentence TTS for better accuracy...")
                    
                    sentence_result = tts_processor.text_to_speech_by_sentences(
                        result['ssml_clean'],
                        gender=args.gender[-1],
                        voice_id=voice_id,
                        output_dir=os.path.dirname(output_mp3_path),
                        base_name=f"sentence_{timestamp}",
                        play_individual=False
                    )
                    
                    if sentence_result and sentence_result['combined_file']:
                        # Copy the combined file to the expected output path
                        import shutil
                        shutil.copy2(sentence_result['combined_file'], output_mp3_path)
                        mp3_path = output_mp3_path
                        
                        if args.debug:
                            print(f"✅ Generated {len(sentence_result['individual_files'])} individual sentences")
                            print(f"✅ Combined into: {output_mp3_path}")
                    else:
                        mp3_path = None
                else:
                    # STANDARD TTS (single generation)
                    mp3_path = tts_processor.text_to_speech(
                        result['ssml_clean'],
                        gender=args.gender[-1],
                        voice_id=voice_id,
                        output_path=output_mp3_path,
                        play_audio=True  # Play single line audio immediately
                    )
                processing_time = (time.time() - start_time) * 1000
                
                if mp3_path:
                    print(f"🎵 Single line audio created: {output_mp3_path}")
                else:
                    print(f"❌ Failed to generate TTS audio")

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

    # Excel export - ALWAYS export table
    try:
        # Export table from single line processing
        if 'processing_table' in result and result['processing_table']:
            table = result['processing_table']
            success = table.export_to_excel("hamara_table.xlsx")
            if success:
                print("📊 Processing table exported to: hamara_table.xlsx")
            else:
                print("❌ Failed to export processing table to Excel")
        # Export table from multi-line processing
        elif 'processed_lines' in result and result['processed_lines']:
            # For multi-line processing, export the first table as example
            # (could be enhanced to combine all tables or export separately)
            first_line_result = result['processed_lines'][0]
            if 'processing_table' in first_line_result and first_line_result['processing_table']:
                table = first_line_result['processing_table']
                success = table.export_to_excel("hamara_table_line1.xlsx")
                if success:
                    print("📊 First line processing table exported to: hamara_table_line1.xlsx")
                    print("💡 Note: Multi-line processing - only first line table exported")
                else:
                    print("❌ Failed to export processing table to Excel")
            else:
                print("❌ No processing table found in results")
        else:
            print("❌ No processing table available for export")
    except Exception as e:
        print(f"❌ Excel export error: {e}")
    
    # Additional Excel export if --excel flag is used
    if args.excel and args.debug:
        print("💡 Note: Table was already exported above. --excel flag is now optional.")

    # Cleanup debug file
    if args.debug:
        # Note: disable_debug_file() was removed - debug file cleanup handled automatically
        print("[OK] Debug output saved to debug.txt")

    # Print parameters summary at the end (read from extract_params.bu if available)
    try:
        import json
        if os.path.exists('extract_params.bu'):
            with open('extract_params.bu', 'r', encoding='utf-8') as f:
                params = json.load(f)
                start_line = params.get('start_line', 1080)
                total_lines = params.get('total_lines', len(result['processed_lines']) if process_line_by_line and 'processed_lines' in result else 1)
                from_line = params.get('from_line', 7)
                show_lines = params.get('show_lines', 1)
                print(f"Parameters: start_line={start_line}, total_lines={total_lines}, from_line={from_line}, show_lines={show_lines}")
        else:
            # Fallback to default values if no backup file exists
            if process_line_by_line and 'processed_lines' in result:
                total_lines = len(result['processed_lines'])
                print(f"Parameters: start_line=1080, total_lines={total_lines}, from_line=7, show_lines=1")
            else:
                print(f"Parameters: start_line=1080, total_lines=1, from_line=7, show_lines=1")
    except Exception:
        # Fallback to default values if reading backup fails
        if process_line_by_line and 'processed_lines' in result:
            total_lines = len(result['processed_lines'])
            print(f"Parameters: start_line=1080, total_lines={total_lines}, from_line=7, show_lines=1")
        else:
            print(f"Parameters: start_line=1080, total_lines=1, from_line=7, show_lines=1")

    return result['ssml_text']

if __name__ == '__main__':
    result = main()
