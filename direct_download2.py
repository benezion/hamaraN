import time
import json
import os
import requests  # Import the requests library
import re  # Import regular expressions module for cleaning SSML tags
import csv  # Import CSV module for reading the client IDs file
import codecs  # For handling different encodings
import argparse  # For command line arguments
import shutil  # For copying files



def remove_ssml_tags(text):
    """Remove SSML tags from text while preserving the content."""
    # For SSML tags with the alias attribute, replace with the alias value
    clean_text = re.sub(r'<say-as[^>]*?alias="([^"]*)"[^>]*?>.*?</say-as>', r'\1', text)
    # For other SSML tags, replace with their content
    clean_text = re.sub(r'<say-as[^>]*?>(.*?)</say-as>', r'\1', clean_text)
    # Remove any other XML tags but keep their content
    clean_text = re.sub(r'<[^>]*?>', '', clean_text)
    return clean_text

def merge_speak_with_ssml(speak_content, all_hebrew_lines):
    """
    Merge <speak> content with <sub alias> SSML tags from corresponding lines.
    This fixes the TTS pronunciation bug by combining speak blocks with proper pronunciation guides.
    """
    if '<speak>' not in speak_content:
        return None

    try:
        # Extract the text content from the <speak> block (without <say-as> content)
        speak_match = re.search(r'<speak>\s*(.*?)\s*(?:<say-as>|</speak>)', speak_content, re.DOTALL)
        if not speak_match:
            return None

        speak_text = speak_match.group(1).strip()

        # Find corresponding line with <sub alias> tags that has similar content
        clean_speak_text = remove_ssml_tags(speak_text)

        best_match = None
        best_match_score = 0

        for hebrew_line in all_hebrew_lines:
            if '<sub alias=' in hebrew_line and '<speak>' not in hebrew_line:
                clean_hebrew = remove_ssml_tags(hebrew_line)

                # Calculate similarity (simple word overlap)
                speak_words = set(clean_speak_text.split())
                hebrew_words = set(clean_hebrew.split())

                if speak_words and hebrew_words:
                    overlap = len(speak_words.intersection(hebrew_words))
                    total = len(speak_words.union(hebrew_words))
                    score = overlap / total if total > 0 else 0

                    if score > best_match_score and score > 0.7:  # 70% similarity threshold
                        best_match = hebrew_line
                        best_match_score = score

        if best_match:
            # Create merged line: <speak> + content with <sub alias> tags + </speak>
            merged_content = f"<speak>{best_match}</speak>"
            return merged_content
        else:
            # If no good match found, return the original speak content
            return speak_content

    except Exception as e:
        print(f"⚠️  Error merging SSML: {e}")
        return speak_content

def read_csv_safely(file_path):
    """Try to read a CSV file with multiple encodings."""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            print(f"Trying to read CSV with {encoding} encoding...")
            rows = []
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f)
                rows = list(reader)
            print(f"Successfully read CSV with {encoding} encoding.")
            return rows
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding")

    # If all encodings fail, try a binary read approach
    try:
        print("Trying binary read approach...")
        with open(file_path, 'rb') as f:
            content = f.read()
            # Replace or remove problematic bytes
            content = content.replace(b'\x9c', b' ')  # Replace problematic byte with space
            text = content.decode('utf-8', errors='ignore')

            # Process as CSV
            import io
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            print("Successfully read CSV with binary approach")
            return rows
    except Exception as e:
        print(f"Binary approach failed: {e}")

    raise Exception(f"Could not read CSV file {file_path} with any encoding")

def create_all_lines_file(original_file, output_file):
    """
    Create a new file containing ONLY lines with "Before tts:" from the original file, with line numbers.
    Extracts content after "Before tts:".
    If content contains vertical bars (|), extract only text after the last bar.
    """
    try:
        if not os.path.exists(original_file):
            print(f"Original file {original_file} not found. Cannot create {output_file}.")
            return

        print(f"\nProcessing {original_file} to create {output_file}...")

        # Read all content from the original file
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Read {len(content)} characters from {original_file}")

        if not content.strip():
            print(f"Warning: {original_file} appears to be empty or contains only whitespace.")
            return

        # Read all content lines from the original file
        content_lines = []

        for line in content.split('\n'):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Only process lines containing "Before tts:"
            if "Before tts:" in line:
                # Extract content after "Before tts:"
                extracted_content = line.split("Before tts:", 1)[1].strip()

                # If the content contains vertical bars, extract only text after the last one
                if '|' in extracted_content:
                    extracted_content = extracted_content.split('|')[-1].strip()

                # Skip lines that contain only numbers after processing
                if not extracted_content.isdigit() and extracted_content:
                    content_lines.append(extracted_content)

        print(f"Extracted {len(content_lines)} 'Before tts:' lines")

        if not content_lines:
            print(f"Warning: No valid 'Before tts:' content could be extracted from {original_file}.")
            # Create an empty file as a fallback
            with open(output_file, 'w', encoding='utf-8') as f:
                pass
            print(f"Warning: Created empty {output_file} since no valid content was found")
            return

        # Write numbered content to the new file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, content in enumerate(content_lines, 1):
                f.write(f"{i:04d} {content}\n")

        print(f"Created {output_file} with {len(content_lines)} numbered lines")

    except Exception as e:
        print(f"Error creating {output_file}: {e}")
        # Make sure we create at least an empty file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass
            print(f"Created empty {output_file} due to error")
        except:
            pass

def split_into_sentences(text):
    """Split text into sentences ending with ',', '.', '?', or '!'"""
    # Match sentences ending with comma, period, question mark, or exclamation mark
    pattern = r'[^,.!?]+[,.!?]'
    sentences = re.findall(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def find_common_short_phrases(text):
    """Find common Hebrew short phrases and greetings"""
    # Regular expression for common short phrases like "שלום," with punctuation
    short_phrases_pattern = r'\b\w+[,.!?]'
    matches = re.findall(short_phrases_pattern, text)

    # Add specific handling for Hebrew greetings
    hebrew_greetings = ["שלום", "היי", "בוקר", "ערב", "לילה"]

    phrases = []
    # Add matches from regex
    phrases.extend([m.strip() for m in matches if m.strip()])

    # Add specific checks for Hebrew greetings with punctuation
    for greeting in hebrew_greetings:
        for punct in [",", ".", "!", "?"]:
            check_phrase = f"{greeting}{punct}"
            if check_phrase in text:
                phrases.append(check_phrase)

    return phrases

def remove_duplicate_sentences(file_path):
    """Process file to remove duplicate sentences and phrases"""
    print(f"\nRemoving duplicate sentences from: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False

    # Set to store unique content
    unique_content = set()

    # Read the original file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Create a new file for the processed content
    new_file_path = f"{file_path}.deduplicated"
    with open(new_file_path, 'w', encoding='utf-8') as f:

        # First pass - collect all unique content
        for line in lines:
            if re.search(r'[\u05D0-\u05EA]', line):  # Only process lines with Hebrew
                # Add sentences
                for sentence in split_into_sentences(line):
                    unique_content.add(sentence)

                # Add short phrases/greetings
                for phrase in find_common_short_phrases(line):
                    unique_content.add(phrase)

                # Add longer phrases
                words = line.split()
                if len(words) >= 3:
                    for i in range(len(words)-2):
                        for j in range(i+3, min(len(words)+1, i+10)):
                            phrase = ' '.join(words[i:j])
                            if len(phrase) > 5:
                                unique_content.add(phrase)

        # Second pass - remove duplicates
        for line_num, line in enumerate(lines, 1):
            original_line = line

            # Skip lines that don't contain Hebrew text
            if not re.search(r'[\u05D0-\u05EA]', line):
                # Check if line only contains numbers after stripping
                stripped_line = original_line.strip()
                if stripped_line.isdigit():
                    print(f"Line {line_num}: Removing line with only numbers: {stripped_line}")
                    continue
                f.write(original_line)
                continue

            modified_line = line
            first_occurrence = {}  # Track if this is the first time we're seeing a phrase

            # First check sentences
            for sentence in split_into_sentences(modified_line):
                if sentence in first_occurrence:
                    print(f"Line {line_num}: Removing duplicate sentence: {sentence}")
                    modified_line = modified_line.replace(sentence, "")
                else:
                    first_occurrence[sentence] = True

            # Then check short phrases and greetings
            for phrase in find_common_short_phrases(modified_line):
                if phrase in first_occurrence:
                    print(f"Line {line_num}: Removing duplicate short phrase: {phrase}")
                    modified_line = modified_line.replace(phrase, "")
                else:
                    first_occurrence[phrase] = True

            # Then check longer phrases (3+ words)
            words = modified_line.split()
            if len(words) >= 3:
                for i in range(len(words)-2):
                    for j in range(i+3, min(len(words)+1, i+10)):
                        phrase = ' '.join(words[i:j])
                        if len(phrase) > 5:
                            if phrase in first_occurrence:
                                print(f"Line {line_num}: Removing duplicate longer phrase: {phrase}")
                                modified_line = modified_line.replace(phrase, "")
                            else:
                                first_occurrence[phrase] = True

            # Remove any double spaces created by removal
            modified_line = re.sub(r' +', ' ', modified_line).strip()

            # Write the modified line if it's not empty and doesn't only contain numbers
            if modified_line and not modified_line.isdigit():
                f.write(modified_line + '\n')
            elif modified_line:
                print(f"Line {line_num}: Removing line with only numbers after cleaning: {modified_line}")

    # Replace the original file with the new file
    try:
        # First try direct replacement
        os.replace(new_file_path, file_path)
    except PermissionError:
        try:
            # If that fails, try to copy the content and rewrite the original file
            print(f"Permission error while replacing file. Trying alternative method...")
            with open(new_file_path, 'r', encoding='utf-8') as src:
                content = src.read()

            with open(file_path, 'w', encoding='utf-8') as dst:
                dst.write(content)

            # Remove the deduplicated file
            try:
                os.remove(new_file_path)
            except:
                print(f"Note: Could not remove temporary file {new_file_path}")
        except Exception as e:
            print(f"Error replacing file: {e}")
            print(f"Deduplicated content is available in: {new_file_path}")
            return False

    print(f"Duplicate content removed successfully from {file_path}")
    print(f"Total unique items found: {len(unique_content)}")
    return True

def create_long_lines(input_file, output_file, max_chars=400):
    """
    Read all_lines.txt and create long lines by combining sentences up to max_chars.
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Read {len(content)} characters from {input_file}")

        if not content.strip():
            print(f"Warning: {input_file} appears to be empty or contains only whitespace.")
            return

        # Find all lines that contain Hebrew characters
        lines = content.split('\n')
        hebrew_lines = []

        for line in lines:
            # Check if line contains Hebrew characters
            if re.search(r'[\u05D0-\u05EA]', line):
                hebrew_lines.append(line.strip())

        print(f"Found {len(hebrew_lines)} lines containing Hebrew text.")

        if not hebrew_lines:
            print("No Hebrew text found.")
            return

        # Extract just the text content (remove line numbers)
        sentences = []
        for line in hebrew_lines:
            # Remove line numbers like "0001 " from the beginning
            cleaned_line = re.sub(r'^\d+\s+', '', line)
            if cleaned_line.strip():
                sentences.append(cleaned_line.strip())

        print(f"Extracted {len(sentences)} individual sentences.")

        if not sentences:
            print("No sentences found after cleaning.")
            return

        # Combine sentences up to max_chars
        combined_sentences = []
        current_combination = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed max_chars
            potential_combination = current_combination + (" " if current_combination else "") + sentence

            if len(potential_combination) <= max_chars:
                current_combination = potential_combination
            else:
                # Save current combination and start new one
                if current_combination:
                    combined_sentences.append(current_combination)
                current_combination = sentence

        # Don't forget the last combination
        if current_combination:
            combined_sentences.append(current_combination)

        print(f"Created {len(combined_sentences)} combined sentences.")

        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(combined_sentences, 1):
                f.write(f"{i}: {sentence}\n")

        print(f"Long sentences written to {output_file}.")

    except Exception as e:
        print(f"Error creating long lines: {e}")

def organize_individual_conversation_files():
    """
    Create individual numbered files for each conversation JSON file found.
    Only write new/unique lines that haven't appeared in previous files.
    """
    # Create conversation_logs directory if it doesn't exist
    logs_dir = "conversation_logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created directory: {logs_dir}")

    # Find all JSON files with actual content
    json_files = []
    for f in os.listdir('.'):
        if f.startswith('content_') and f.endswith('.json'):
            try:
                # Check file size - only include files larger than 100 bytes (not empty)
                if os.path.getsize(f) > 100:
                    json_files.append(f)
            except:
                pass

    if not json_files:
        print("⚠️  No JSON files found")
        return

    print(f"Found {len(json_files)} conversation JSON files")

    # Sort JSON files by creation time or name for consistent numbering
    json_files.sort()

    # Global tracking sets for all previously seen content
    global_seen_hebrew_lines = set()
    global_seen_tts_lines = set()
    global_seen_clean_lines = set()
    global_seen_long_lines = set()
    global_seen_tts_pairs = set()

    # Process each JSON file
    for i, json_file in enumerate(json_files, 1):
        sequence_num = f"{i:05d}"  # 00001, 00002, etc.

        print(f"\n📁 Processing conversation {sequence_num} from {json_file}")

        # First, copy the JSON file to the organized location
        organized_json_path = os.path.join(logs_dir, f"{sequence_num}_content.json")
        shutil.copy2(json_file, organized_json_path)
        print(f"✅ Saved organized JSON: {organized_json_path}")

        # Then create individual files for this conversation using the organized JSON file
        create_individual_conversation_files(organized_json_path, sequence_num, logs_dir)

def create_individual_conversation_files(json_file, sequence_num, logs_dir):
    """
    Create all the individual files for a single conversation.
    Only write lines that haven't been seen in previous conversations.
    """
    try:
        # Read the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"⚠️  JSON file {json_file} doesn't contain expected list format")
            return

        # Extract conversation ID from filename (handle both organized and original paths)
        filename = os.path.basename(json_file)
        if filename.startswith('00'):
            # This is an organized file like "00001_content.json"
            conversation_id = sequence_num  # Use the sequence number as ID
        else:
            # This is an original file like "content_xxxxx.json"
            conversation_id = filename.replace('content_', '').replace('.json', '')



        # Create files in conversation_logs directory
        base_path = os.path.join(logs_dir, sequence_num)

        # Note: JSON file is already copied to the organized location by the caller

                # 1. Before TTS file (Hebrew text from output_event entries)
        # Extract Hebrew text from output_event entries - this is the original text before TTS processing
        before_tts_lines = []
        for item in data:
            if isinstance(item, dict) and "content" in item and "stage" in item and item["stage"] == "output_event":
                content = item["content"]
                if re.search(r'[\u05D0-\u05EA]', content):  # Check for Hebrew characters
                    # Extract Hebrew text from content - skip JSON parts
                    # Look for Hebrew text after JSON structures
                    if content.startswith('{"'):
                        # Find where JSON ends and Hebrew text begins
                        try:
                            # Find the closing brace and extract text after it
                            json_end = content.rfind('}')
                            if json_end != -1:
                                hebrew_part = content[json_end + 1:].strip()
                                # Clean up any HTML-like tags including </>
                                hebrew_part = re.sub(r'<[^>]*/?>', '', hebrew_part).strip()
                                if hebrew_part and re.search(r'[\u05D0-\u05EA]', hebrew_part):
                                    before_tts_lines.append(hebrew_part)
                        except:
                            # If JSON parsing fails, try to extract Hebrew text directly
                            hebrew_text = re.findall(r'[א-ת\s.,!?:;״""\'()]+', content)
                            for text in hebrew_text:
                                if text.strip() and len(text.strip()) > 2:
                                    before_tts_lines.append(text.strip())
                    else:
                        # If content doesn't start with JSON, it might be pure Hebrew text
                        clean_content = re.sub(r'<[^>]*/?>', '', content.strip()).strip()
                        if clean_content and re.search(r'[\u05D0-\u05EA]', clean_content):
                            before_tts_lines.append(clean_content)

        # DON'T remove duplicates - keep all lines as they appear in conversation order
        # This ensures we get all 5 lines from output_event entries

        # For before_tts files, remove duplicates within the same conversation
        # Keep only unique lines within this specific conversation
        seen_lines = set()
        unique_before_tts_lines = []
        for line in before_tts_lines:
            if line.strip() and line not in seen_lines:
                unique_before_tts_lines.append(line)
                seen_lines.add(line)

        if unique_before_tts_lines:
            before_tts_target = f"{base_path}_before_tts.txt"
            with open(before_tts_target, 'w', encoding='utf-8') as f:
                for line in unique_before_tts_lines:
                    f.write(f"{line}\n")
                    print(f"Before TTS line: {line}")
            print(f"✅ Created: {before_tts_target} with {len(unique_before_tts_lines)} unique lines (text before TTS processing)")

            # 2. Long lines file (created as 1 single long line from all before_tts content)
            if unique_before_tts_lines:
                # Combine all lines into one single long line with spaces
                single_long_line = " ".join(unique_before_tts_lines)

                long_target = f"{base_path}_long_lines.txt"
                with open(long_target, 'w', encoding='utf-8') as f:
                    f.write(f"1: {single_long_line}\n")
                print(f"✅ Created: {long_target} as 1 single long line from all before_tts content ({len(single_long_line)} characters)")
            else:
                print(f"⚠️  No long lines created for {sequence_num}")

        else:
            print(f"⚠️  No before-TTS lines found for {sequence_num}")

    except Exception as e:
        print(f"❌ Error processing {json_file}: {e}")



def create_combined_before_tts_from_individual_files():
    """
    Create a combined all_before_tts.txt file from all individual xxxxx_before_tts.txt files
    in conversation order (00001, 00002, etc.).
    """
    logs_dir = "conversation_logs"
    output_file = "all_before_tts.txt"

    if not os.path.exists(logs_dir):
        print(f"⚠️  Directory {logs_dir} not found. Cannot create combined before_tts file")
        return

    # Find all xxxxx_before_tts.txt files
    before_tts_files = []
    for filename in os.listdir(logs_dir):
        if filename.endswith("_before_tts.txt"):
            before_tts_files.append(filename)

    if not before_tts_files:
        print(f"⚠️  No xxxxx_before_tts.txt files found in {logs_dir}")
        return

    # Sort files by conversation number (00001, 00002, etc.)
    before_tts_files.sort()

    print(f"📁 Found {len(before_tts_files)} individual before_tts files")

    # Combine all content into single file
    combined_lines = []
    for filename in before_tts_files:
        file_path = os.path.join(logs_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Add lines from this conversation (strip newlines but keep content)
                for line in lines:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        combined_lines.append(line)
            print(f"✅ Added {len(lines)} lines from {filename}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # Write combined content to all_before_tts.txt
    if combined_lines:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in combined_lines:
                f.write(f"{line}\n")
        print(f"✅ Created combined {output_file} with {len(combined_lines)} total lines from all conversations")
    else:
        print(f"⚠️  No content found to create {output_file}")

def create_combined_long_lines_from_individual_files():
    """
    Create a combined all_long_lines.txt file from all individual xxxxx_long_lines.txt files
    in conversation order (00001, 00002, etc.) with proper conversation numbering.
    """
    logs_dir = "conversation_logs"
    output_file = "all_long_lines.txt"

    if not os.path.exists(logs_dir):
        print(f"⚠️  Directory {logs_dir} not found. Cannot create combined long_lines file")
        return

    # Find all xxxxx_long_lines.txt files
    long_lines_files = []
    for filename in os.listdir(logs_dir):
        if filename.endswith("_long_lines.txt"):
            long_lines_files.append(filename)

    if not long_lines_files:
        print(f"⚠️  No xxxxx_long_lines.txt files found in {logs_dir}")
        return

    # Sort files by conversation number (00001, 00002, etc.)
    long_lines_files.sort()

    print(f"📁 Found {len(long_lines_files)} individual long_lines files")

    # Combine all content into single file with proper conversation numbering
    combined_lines = []
    conversation_number = 1  # Start numbering from 1

    for filename in long_lines_files:
        file_path = os.path.join(logs_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Process lines from this conversation
                for line in lines:
                    line = line.strip()
                    if line:  # Only add non-empty lines
                        # Remove the existing "1:" prefix and add proper conversation number
                        if line.startswith("1: "):
                            content = line[3:]  # Remove "1: " prefix
                            numbered_line = f"{conversation_number}: {content}"
                            combined_lines.append(numbered_line)
                            conversation_number += 1  # Increment for next conversation
                        else:
                            # If line doesn't start with "1: ", add conversation number anyway
                            numbered_line = f"{conversation_number}: {line}"
                            combined_lines.append(numbered_line)
                            conversation_number += 1
            print(f"✅ Added {len(lines)} lines from {filename}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    # Write combined content to all_long_lines.txt
    if combined_lines:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in combined_lines:
                f.write(f"{line}\n")
        print(f"✅ Created combined {output_file} with {len(combined_lines)} total lines from all conversations")
        print(f"✅ Conversations are numbered from 1 to {conversation_number-1}")
    else:
        print(f"⚠️  No content found to create {output_file}")

def download_content_for_multiple_clients(max_rows=10):
    """Downloads content for multiple clients from a CSV file and saves to a combined file."""

    # CSV file with client IDs
    csv_file_path = "2025-06-17T08-07_export.csv"
    output_file = "full_content_tts.txt"

    print(f"📊 Processing maximum {max_rows} rows from CSV file")

    # Check if output file exists and remove it if needed
    if os.path.exists(output_file):
        print(f"Removing existing output file: {output_file}")
        try:
            os.remove(output_file)
            print(f"Successfully removed existing {output_file}")
            # Reset tracking variables since we're starting fresh
            existing_conv_ids = set()
            existing_content_lines = set()
            last_conv_number = 0
        except Exception as e:
            print(f"Error removing existing file: {e}")
            # Continue with appending to existing file
            # Set to store existing conversation IDs
            existing_conv_ids = set()
            # Set to store existing content lines
            existing_content_lines = set()
            # Variable to track the last conversation number
            last_conv_number = 0

            # Read existing conversation IDs
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract conversation numbers and IDs using regex
                    conv_info = re.findall(r'Conversation (\d+):\s+ID=([^\s\n]+)', content)

                    for num, conv_id in conv_info:
                        existing_conv_ids.add(conv_id)
                        # Update last conversation number
                        conv_num = int(num)
                        if conv_num > last_conv_number:
                            last_conv_number = conv_num

                    # Extract existing content lines
                    content_lines = re.findall(r'1\. Before tts: (.+)', content)
                    for line in content_lines:
                        existing_content_lines.add(line.strip())

                    print(f"Found {len(existing_conv_ids)} existing conversations. Last conversation number: {last_conv_number}")
                    print(f"Found {len(existing_content_lines)} existing content lines")
            except Exception as e:
                print(f"Error reading existing output file: {e}")
                # If there's an error reading the file, proceed with creating a new one
                pass
    else:
        # File doesn't exist, start with fresh tracking variables
        existing_conv_ids = set()
        existing_content_lines = set()
        last_conv_number = 0

    # Counters for tracking humain_utter items
    total_json_items = 0
    humain_utter_items = 0

    try:
        # Read the CSV file with safe approach
        rows = read_csv_safely(csv_file_path)

        # Track which row number from the CSV we're processing
        row_number = 0
        # Count of new conversations added
        new_conversations = 0
        # Count of duplicate content lines skipped
        duplicate_lines_skipped = 0
        # Count of skipped conversations (already exist)
        skipped_conversations = 0

        # Open the output file for appending
        with open(output_file, 'a', encoding='utf-8') as output_f:

            for row in rows:
                if not row:  # Skip empty rows
                    continue

                # Increment row counter for non-empty rows
                row_number += 1

                # Check if we've reached the maximum number of rows to process
                if row_number > max_rows:
                    print(f"\n🛑 Reached maximum limit of {max_rows} rows. Stopping processing.")
                    break

                print(f"\n==== ROW NUMBER: {row_number}/{max_rows} ====")

                # Extract client ID from column A
                client_id_cell = row[0]
                print(f"\nRaw CSV cell value: {client_id_cell}")

                # Specifically look for conversation_id parameter
                match = re.search(r'conversation_id=([^&\s]+)', client_id_cell)
                if not match:
                    print(f"WARNING: Cannot extract conversation_id parameter from cell. Skipping this row.")
                    continue

                conversation_id = match.group(1)  # This extracts the value after conversation_id=

                # Check if this conversation ID is already processed
                if conversation_id in existing_conv_ids:
                    print(f"Conversation ID {conversation_id} already exists in output file. Skipping HTTP request.")
                    skipped_conversations += 1
                    continue

                client_id = "iec:meirachat"  # Updated to production client_id
                print(f"Extracted conversation_id: {conversation_id}")
                print(f"Client ID (updated): {client_id}")

                # Construct URL for display (with @ symbol)
                display_url = f"@https://humains-core.appspot.com/hub/clients/{client_id}/conversations/{conversation_id}"
                print(f"Display URL format:")
                print(f"FULL URL: {display_url}")

                # Construct actual URL for HTTP request (without @ symbol)
                actual_url = f"https://humains-core.appspot.com/hub/clients/{client_id}/conversations/{conversation_id}"
                print(f"Making HTTP request to exact URL format: {actual_url}")

                try:
                    # Make the request with timeout
                    print("Sending HTTP request...")
                    response = requests.get(actual_url, timeout=30)  # 30 second timeout
                    print("Request sent, waiting 2 seconds for server to respond...")
                    time.sleep(2)  # Wait 2 seconds after opening the URL

                    response.raise_for_status()
                    data = response.json()

                    # Check if we have valid data
                    if data:
                        print(f"Data fetched successfully for conversation ID: {conversation_id}! Received {len(data) if isinstance(data, list) else 'some'} data.")

                        # Count total items and humain_utter items for debugging
                        if isinstance(data, list):
                            total_items_in_this_conv = len(data)
                            humain_utter_in_this_conv = sum(1 for item in data if isinstance(item, dict) and "stage" in item and item["stage"] == "humain_utter")
                            total_json_items += total_items_in_this_conv
                            humain_utter_items += humain_utter_in_this_conv
                            print(f"Conversation ID {conversation_id} has {total_items_in_this_conv} total items, {humain_utter_in_this_conv} are humain_utter")
                    else:
                        print(f"Warning: Received empty data for conversation ID: {conversation_id}. Continuing anyway.")

                    # Process conversation data - we only need to save the JSON file
                    # All text processing will be done later from the saved JSON files

                    # Increment conversation number for this new conversation
                    last_conv_number += 1

                    # Save JSON file directly to conversation_logs directory with sequence number
                    sequence_num = f"{last_conv_number:05d}"  # 00001, 00002, etc.
                    logs_dir = "conversation_logs"
                    if not os.path.exists(logs_dir):
                        os.makedirs(logs_dir)

                    json_filename = os.path.join(logs_dir, f"{sequence_num}_content.json")
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"✅ JSON file saved to: {json_filename}")

                    # Add to existing set of conversation IDs
                    existing_conv_ids.add(conversation_id)

                    # Print success message with conversation ID - make it very visible
                    print(f"\n>>> SUCCESSFULLY PROCESSED LINE {row_number}, CONVERSATION ID: {conversation_id} <<<\n")

                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout: Server took longer than 30 seconds to respond for {conversation_id}")
                except requests.exceptions.ConnectionError:
                    print(f"🌐 Connection Error: Could not connect to server for {conversation_id}")
                except requests.exceptions.RequestException as e:
                    print(f"🔴 HTTP Error downloading data for {conversation_id}: {e}")
                except json.JSONDecodeError:
                    print(f"📄 JSON Error: Invalid response format for {conversation_id}")
                except Exception as e:
                    print(f"❌ Unexpected error processing {conversation_id}: {e}")

            print(f"\nAll conversations processed.")
            print(f"Processed {row_number} rows from CSV")
            print(f"Saved {last_conv_number} conversations to conversation_logs directory")
            print(f"Skipped {skipped_conversations} existing conversations (no HTTP request made)")
            print(f"Skipped {duplicate_lines_skipped} duplicate content lines")
            print(f"Processed {total_json_items} total JSON items, of which {humain_utter_items} were humain_utter items")





        # Process each saved JSON file in conversation_logs to create individual files
        logs_dir = "conversation_logs"
        if os.path.exists(logs_dir):
            # Find all nnnnn_content.json files
            json_files = []
            for filename in os.listdir(logs_dir):
                if filename.endswith("_content.json") and filename[:5].isdigit():
                    json_files.append(filename)

            if json_files:
                # Sort files by sequence number
                json_files.sort()



                print(f"\n📁 Processing {len(json_files)} saved conversation files...")

                # Process each JSON file
                for json_file in json_files:
                    sequence_num = json_file[:5]  # Extract 00001, 00002, etc.
                    json_path = os.path.join(logs_dir, json_file)

                    print(f"\n📁 Processing conversation {sequence_num} from {json_file}")

                    # Create individual files for this conversation
                    create_individual_conversation_files(json_path, sequence_num, logs_dir)

        # Create combined files from individual conversation files
        print("\n📝 Creating combined files from all conversations...")
        create_combined_before_tts_from_individual_files()
        create_combined_long_lines_from_individual_files()

        print("\nAll processing steps completed successfully.")

    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Download conversation content from CSV file')
    parser.add_argument('max_rows', nargs='?', type=int, default=10,
                       help='Maximum number of CSV rows to process (default: 10)')

    args = parser.parse_args()

    print(f"🚀 Starting conversation download with limit: {args.max_rows} rows")
    download_content_for_multiple_clients(args.max_rows)

if __name__ == "__main__":
    main()
