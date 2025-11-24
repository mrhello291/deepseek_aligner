import json
import re

def find_suspicious_samples(input_file, suspicious_file):
    """
    Reads a JSONL file and filters out suspicious samples
    based on simple string-matching heuristics.
    """
    
    # We will write suspicious samples here
    suspicious_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                
                # Make text lowercase and strip whitespace for better matching
                ideal = data.get('ideal_answer', '').lower().strip()
                fake = data.get('fake_answer', '').lower().strip()

                # --- Heuristics to find "correct" fake answers ---
                
                # 1. The fake answer is identical to the ideal answer.
                if ideal == fake:
                    data['reason_flagged'] = 'Identical'
                    suspicious_entries.append(data)
                    continue # No need to check other rules

                # 2. The fake answer *contains* the ideal answer.
                # (This catches cases where fake is a full-sentence 
                # version of the ideal answer)
                # We check if ideal is not empty and is long enough
                # to be meaningful (e.g., > 3 chars)
                if ideal and len(ideal) > 3 and ideal in fake:
                    data['reason_flagged'] = 'FakeContainsIdeal'
                    suspicious_entries.append(data)
                    continue
                    
                # 3. The ideal answer *contains* the fake answer.
                # (This catches the reverse case)
                if fake and len(fake) > 3 and fake in ideal:
                    data['reason_flagged'] = 'IdealContainsFake'
                    suspicious_entries.append(data)
                    continue

            except json.JSONDecodeError:
                print(f"Skipping bad line: {line}")
    
    # Write all suspicious entries to a new file
    with open(suspicious_file, 'w', encoding='utf-8') as outfile:
        for entry in suspicious_entries:
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"Found {len(suspicious_entries)} suspicious samples.")
    print(f"Review them in: {suspicious_file}")

# --- HOW TO USE ---
# 1. Make sure you have your 25k samples in a file, e.g., 'all_data.jsonl'
# 2. Run this script.
find_suspicious_samples('cleaned_hard_fakes.jsonl', 'suspicious_samples.jsonl')