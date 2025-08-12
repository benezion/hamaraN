import time
import json
import os
import requests  # Import the requests library
import re  # Import regular expressions module for cleaning SSML tags
import csv  # Import CSV module for reading the client IDs file
import codecs  # For handling different encodings
import hashlib  # For creating hash keys for the cache

# Global TTS cache to prevent reprocessing the same text
TTS_CACHE = {}
TTS_CACHE_FILE = "tts_cache.json"

# Load existing TTS cache if available
def load_tts_cache():
    global TTS_CACHE
    if os.path.exists(TTS_CACHE_FILE):
        try:
            with open(TTS_CACHE_FILE, 'r', encoding='utf-8') as f:
                TTS_CACHE = json.load(f)
            print(f"Loaded {len(TTS_CACHE)} cached TTS entries from {TTS_CACHE_FILE}")
        except Exception as e:
            print(f"Error loading TTS cache: {e}")
            TTS_CACHE = {}
    else:
        TTS_CACHE = {}

# Save TTS cache to file
def save_tts_cache():
    if TTS_CACHE:
        try:
            with open(TTS_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(TTS_CACHE, f, ensure_ascii=False)
            print(f"Saved {len(TTS_CACHE)} entries to TTS cache file")
        except Exception as e:
            print(f"Error saving TTS cache: {e}")

# Function to get TTS for text with caching
def get_tts_with_cache(text):
    # Create a hash key for the text
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    # Check if this text is already in the cache
    if text_hash in TTS_CACHE:
        print(f"TTS cache hit for: {text[:50]}...")
        return TTS_CACHE[text_hash]

    # Not in cache, so we need to process it
    # This is a placeholder - you would insert your actual TTS processing code here
    print(f"DEBUG: Plain text for TTS: {text[:100]}...")
    # Simulate TTS processing time for demonstration purposes
    start_time = time.time()

    # *** Add your actual TTS processing code here ***
    # For example:
    # tts_result = your_tts_engine.process(text)

    # For now, we'll just create a placeholder result
    tts_result = f"<speak>{text}</speak>"

    # Simulate TTS processing delay - remove this in real implementation
    time.sleep(0.5)  # Much faster than 43 seconds

    elapsed_time = time.time() - start_time
    print(f"INFO: TTS processing completed in {elapsed_time:.2f} seconds")

    # Store in cache
    TTS_CACHE[text_hash] = tts_result

    # Save cache periodically (every 10 new entries)
    if len(TTS_CACHE) % 10 == 0:
        save_tts_cache()

    return tts_result

def remove_ssml_tags(text):
    """Remove SSML tags from text while preserving the content."""
    # For SSML tags with the alias attribute, replace with the alias value
    clean_text = re.sub(r'<say-as[^>]*?alias="([^"]*)"[^>]*?>.*?</say-as>', r'\1', text)
    # For other SSML tags, replace with their content
    clean_text = re.sub(r'<say-as[^>]*?>(.*?)</say-as>', r'\1', clean_text)
    # Remove any other XML tags but keep their content
    clean_text = re.sub(r'<[^>]*?>', '', clean_text)
    return clean_text

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
    Combine sentences from input_file into longer sentences up to max_chars characters
    and write them to output_file.
    Preserves punctuation marks (.,!?) at the end of each sentence.
    Removes input line numbers (like 0001, 0012, etc.) from the text.
    """
    print(f"\nCreating long lines (up to {max_chars} characters) from {input_file}...")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return False

    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Read {len(content)} characters from {input_file}")

    if not content.strip():
        print(f"Warning: {input_file} appears to be empty or contains only whitespace.")
        return False

    # Count Hebrew content lines
    hebrew_line_count = 0
    for line in content.split('\n'):
        if line.strip() and re.search(r'[\u05D0-\u05EA]', line):
            hebrew_line_count += 1

    print(f"Found {hebrew_line_count} lines containing Hebrew text.")

    # Extract individual sentences more generously
    sentences = []
    pattern = r'[^,.!?]+[,.!?]'

    # Try first with punctuation-based pattern
    for line in content.split('\n'):
        if line.strip() and re.search(r'[\u05D0-\u05EA]', line):  # Only process non-empty lines with Hebrew
            # First, remove line numbering prefix (e.g., "0123 ")
            clean_line = re.sub(r'^\d+\s+', '', line.strip())
            # Also remove any 4-digit numbers followed by a space anywhere in the text (like "0012 ")
            clean_line = re.sub(r'\b\d{4}\s+', '', clean_line)

            # Try to extract sentences with punctuation
            line_sentences = re.findall(pattern, clean_line)

            # If no sentences with punctuation, treat the whole line as a sentence
            if not line_sentences and not clean_line.strip().isdigit():
                # Ensure the line ends with punctuation
                if clean_line and re.search(r'[\u05D0-\u05EA]', clean_line):
                    if not clean_line[-1] in ['.', ',', '!', '?']:
                        clean_line += '.'  # Add period if no punctuation
                    sentences.append(clean_line)
            else:
                sentences.extend([s.strip() for s in line_sentences if s.strip()])

    print(f"Extracted {len(sentences)} individual sentences.")

    if not sentences:
        print(f"Warning: No valid sentences could be extracted from {input_file}.")
        print("Trying alternate method by using whole lines...")

        # Backup approach: just use whole lines that have Hebrew
        for line in content.split('\n'):
            if line.strip() and re.search(r'[\u05D0-\u05EA]', line) and not line.strip().isdigit():
                # Remove any line numbering prefix and 4-digit numbers
                clean_line = re.sub(r'^\d+\s+', '', line.strip())
                clean_line = re.sub(r'\b\d{4}\s+', '', clean_line)

                if clean_line:
                    # Ensure the line ends with punctuation
                    if not clean_line[-1] in ['.', ',', '!', '?']:
                        clean_line += '.'  # Add period if no punctuation
                    sentences.append(clean_line)

        print(f"Alternate method found {len(sentences)} lines.")

    if not sentences:
        print(f"Error: Still could not extract any valid content from {input_file}.")
        # Create an empty output file so at least it exists
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")
        return False

    # Combine sentences into longer sentences
    long_sentences = []
    current_sentence = ""
    current_length = 0

    for sentence in sentences:
        # Remove any remaining 4-digit numbers followed by space
        sentence = re.sub(r'\b\d{4}\s+', '', sentence)

        # Make sure the sentence ends with punctuation
        if not sentence[-1] in ['.', ',', '!', '?']:
            sentence += '.'

        # If adding this sentence would exceed max_chars
        if current_length + len(sentence) + 1 > max_chars:  # +1 for space
            if current_sentence:  # Only add if we have something
                long_sentences.append(current_sentence.strip())
            current_sentence = sentence
            current_length = len(sentence)
        else:
            # Add this sentence to the current long sentence
            if current_sentence:
                # Ensure proper spacing - add a space after punctuation
                current_sentence += " " + sentence
            else:
                current_sentence = sentence
            current_length = len(current_sentence)

    # Add the last sentence if there's anything left
    if current_sentence:
        long_sentences.append(current_sentence.strip())

    print(f"Created {len(long_sentences)} combined sentences.")

    # Write the long sentences to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(long_sentences, 1):
            f.write(f"{i}: {sentence}\n\n")

    print(f"Long sentences written to {output_file}.")
    return True

def download_content_for_multiple_clients():
    """Downloads content for multiple clients from a CSV file and saves to a combined file."""

    # Load TTS cache at the start
    load_tts_cache()

    # CSV file with client IDs
    csv_file_path = r"d:\benezion\magic94g\bot\chashmal\clients_id_04_06_2025.csv"
    output_file = "full_content_tts.txt"

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

                print(f"\n==== ROW NUMBER: {row_number} ====")

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

                client_id = "app:uti"  # Using the same client_id as before
                print(f"Extracted conversation_id: {conversation_id}")
                print(f"Client ID (fixed): {client_id}")

                # Construct URL for display (with @ symbol)
                display_url = f"@https://humains-core-dev.appspot.com/hub/clients/{client_id}/conversations/{conversation_id}"
                print(f"Display URL format:")
                print(f"FULL URL: {display_url}")

                # Construct actual URL for HTTP request (without @ symbol)
                actual_url = f"https://humains-core-dev.appspot.com/hub/clients/{client_id}/conversations/{conversation_id}"
                print(f"Making HTTP request to exact URL format: {actual_url}")

                try:
                    # Make the request
                    response = requests.get(actual_url)
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

                    # Process conversation data
                    original_hebrew_data = []
                    tts_converted_data = []

                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "content" in item and "stage" in item and item["stage"] == "humain_utter":
                                content = item["content"]
                                original_hebrew_data.append(content)
                                # Use the caching system for TTS conversion
                                tts_converted_data.append(get_tts_with_cache(content))
                                print(f"Found humain_utter: {content[:50]}...")
                            elif isinstance(item, dict) and "stage" in item:
                                print(f"Skipping item with stage: {item['stage']}")
                            elif isinstance(item, dict):
                                print(f"Skipping item without stage field")

                    # Save individual JSON file for this conversation
                    json_filename = f"content_{conversation_id}.json"
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"JSON file saved to: {json_filename}")

                    # Filter the data to include only Hebrew content
                    filtered_hebrew_data = []
                    filtered_tts_data = []

                    for i in range(len(original_hebrew_data)):
                        original_content = original_hebrew_data[i]

                        # Check if content contains Hebrew characters
                        has_hebrew = bool(re.search(r'[\u05D0-\u05EA]', original_content))

                        # Only keep content with Hebrew characters
                        if has_hebrew:
                            # Clean SSML tags
                            cleaned_content = remove_ssml_tags(original_content)
                            filtered_hebrew_data.append(cleaned_content)
                            filtered_tts_data.append(tts_converted_data[i])

                    # Check if we have any new content to add
                    has_new_content = False
                    for cleaned_content in filtered_hebrew_data:
                        if cleaned_content.strip() not in existing_content_lines:
                            has_new_content = True
                            break

                    # Increment conversation number for this new conversation
                    last_conv_number += 1
                    new_conversations += 1

                    # Write new formatted header for this conversation
                    output_f.write(f"Conversation {last_conv_number}:    ID={conversation_id}\n")
                    output_f.write(f"-----------------\n")

                    if not has_new_content:
                        print(f"All content in conversation {conversation_id} already exists in output file. Only adding conversation ID.")
                        # Add two blank lines after conversation header
                        output_f.write("\n\n")
                        # Add to existing set of conversation IDs
                        existing_conv_ids.add(conversation_id)
                        continue

                    # Now write the filtered data with proper formatting
                    new_content_added = False

                    for i in range(len(filtered_hebrew_data)):
                        cleaned_content = filtered_hebrew_data[i]

                        # Check if this specific content already exists
                        if cleaned_content.strip() in existing_content_lines:
                            print(f"Skipping duplicate content: {cleaned_content[:50]}...")
                            duplicate_lines_skipped += 1
                            continue

                        # Add blank line before each pair if content was added before
                        if new_content_added:
                            output_f.write("\n")

                        # Write the content pair
                        output_f.write(f"1. Before tts: {cleaned_content}\n")
                        output_f.write(f"2. After tts: {filtered_tts_data[i]}\n")

                        # Add to existing content lines
                        existing_content_lines.add(cleaned_content.strip())
                        new_content_added = True

                    # Add two blank lines after each conversation with content
                    if new_content_added:
                        output_f.write("\n\n")

                    # Add to existing set of conversation IDs
                    existing_conv_ids.add(conversation_id)

                    # Print success message with conversation ID - make it very visible
                    print(f"\n>>> SUCCESSFULLY PROCESSED LINE {row_number}, CONVERSATION ID: {conversation_id} <<<\n")

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading data for {conversation_id}: {e}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON response for {conversation_id}. The URL might not be returning valid JSON.")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {conversation_id}: {e}")

            print(f"\nAll conversations processed.")
            print(f"Added {new_conversations} new conversations to {output_file}")
            print(f"Skipped {skipped_conversations} existing conversations (no HTTP request made)")
            print(f"Skipped {duplicate_lines_skipped} duplicate content lines")
            print(f"Processed {total_json_items} total JSON items, of which {humain_utter_items} were humain_utter items")

        # Create all_lines.txt with numbered content
        create_all_lines_file(output_file, "all_lines.txt")

        # Remove duplicate sentences from the files
        remove_duplicate_sentences("full_content_tts.txt")
        remove_duplicate_sentences("all_lines.txt")

        # Create long lines file
        create_long_lines("all_lines.txt", "long_lines.txt")

        # Save the final TTS cache
        save_tts_cache()

        print("\nAll processing steps completed successfully.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_content_for_multiple_clients()
