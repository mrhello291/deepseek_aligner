import json
import re

def clean_fake_answers(input_file, output_file):
    """
    Reads a JSONL file, cleans the 'fake_answer' field, 
    and writes the cleaned data to a new JSONL file.
    """
    
    # This regex pattern looks for the first occurrence of any of the
    # explanatory phrases.
    
    # --- THIS LINE IS FIXED ---
    # I added the closing parenthesis ')' at the very end of the pattern
    # to close the main group.
    stop_phrases_pattern = re.compile(
        r'(\n \nNote:|\n\nNote:|\s\(Note:|\n\n\(Note:'
        r'|\n\nExplanation:|\n\nThe flawed answer'
        r'|\n\(Answer generated|\n\nFlawed Causal Answer:'
        r'|\n\n\[Note:|\n\n\(Answer:)'  # <-- The closing ')' was added here
    )

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    data = json.loads(line)
                    
                    if 'fake_answer' in data:
                        original_answer = data['fake_answer']
                        
                        # Split the string at the very first match of our pattern
                        # and take the first part [0]
                        parts = stop_phrases_pattern.split(original_answer, 1)
                        cleaned_answer = parts[0]
                        
                        # Remove any trailing newlines or spaces
                        data['fake_answer'] = cleaned_answer.strip()
                    
                    # Write the cleaned data to the new file
                    json.dump(data, outfile)
                    outfile.write('\n')
                
                except json.JSONDecodeError:
                    print(f"Skipping bad line: {line}")

        print(f"Successfully cleaned data and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- How to use this ---

# 1. Save the code above as a Python file 
#    (e.g., 'hardfakeclearner.py').
# 2. Make sure your data file ('hard_fakes.jsonl') is in the same directory.
# 3. Run the script.

# Example usage (matching your script):
clean_fake_answers('hard_fakes.jsonl', 'cleaned_hard_fakes.jsonl')