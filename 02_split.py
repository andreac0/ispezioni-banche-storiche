import os
import re
import shutil
def is_repetitive(text, threshold=0.2):
    """
    Checks if a page is mostly repetitive garbage.
    Returns True if the ratio of unique words to total words is too low.
    """
    words = text.split()
    if len(words) < 10:  # Don't filter very short pages
        return False
    
    unique_words = set(words)
    ratio = len(unique_words) / len(words)
    
    # If unique words are less than 20% of the total, it's likely a hallucination loop
    return ratio < threshold
import re

def parse_pages(file_path):
    """Parses the file into a list of pages with metadata and content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # This allows for the colon after FILE, handles spaces, and captures the filename
    page_pattern = r"={5,}\s*(?:FILE|PAGE):\s*(.*?)\s*={5,}"
    parts = re.split(page_pattern, content)
    
    page_list = []
    
    # FALLBACK: If no markers were found, treat the entire file as one page
    if len(parts) == 1:
        content_actual = parts[0].strip()
        if content_actual: # Only add if the file isn't completely empty
            page_list.append({
                'num': 1,
                'metadata': {},
                'content': content_actual
            })
        return page_list

    # Original loop for files that DO have headers
    for i in range(1, len(parts), 2):
        page_id = parts[i]
        page_raw = parts[i+1].strip()
        
        try:
            page_num = int(re.search(r'(\d+)', page_id).group(1))
        except (AttributeError, ValueError):
            page_num = i // 2 

        meta_match = re.match(r"-{2,3}(.*?)-{2,3}(.*)", page_raw, re.DOTALL)
        if meta_match:
            metadata_raw = meta_match.group(1).strip()
            content_actual = meta_match.group(2).strip()
            
            metadata = {}
            for line in metadata_raw.split('\n'):
                if ':' in line:
                    k, v = line.split(':', 1)
                    metadata[k.strip()] = v.strip()
        else:
            content_actual = page_raw
            metadata = {}

        if is_repetitive(content_actual):
            content_actual = "(OMISS)"
            
        page_list.append({
            'num': int(page_num),
            'metadata': metadata,
            'content': content_actual
        })
        
    return page_list
    
def split_into_letters(pages):
    """Groups pages into individual letters based on heuristics."""
    letters = []
    current_letter_pages = []
    
    # Heuristics for closings (end of letter)
    closing_patterns = [
        r"distinti\s+ossequi",
        r"distinti\s+saluti",
        r"ossequi\b",
        r"migliori\s+saluti",
        r"cordiali\s+saluti",
        r"con\s+osservanza",
        r"devot[io]\s+ossequi",
        # r"IL\s+DIRETTORE",
        r"Per\s+delegazione",
        r"Direzione\s+Generale\s*$",
        r"In\s+attesa\s+di\s+pregiate\s+comunicazioni",
        r"pongo\s+deferenti\s+ossequi",
        r"porgo\s+ossequi"
        r"la\s+riverisco",
        r"distinti\s+essequi"
    ]
    closing_regex = re.compile("|".join(closing_patterns), re.IGNORECASE)

    # Heuristics for letterheads (start of new letter)
    # We look for place/date patterns or explicit "OGGETTO"
    start_patterns = [
        r"^OGGETTO\s*",
        r"^Oggetto\s*:",
        r"^\*200000\d+\*" # Archive codes often start a new file/document
    ]
    start_regex = re.compile("|".join(start_patterns), re.IGNORECASE | re.MULTILINE)

    for page in pages:
        content = page['content']
        
        # If we have an ongoing letter, check if this page looks like a fresh start
        if current_letter_pages:
            # We look at the first 300 chars for a strong start marker
            if start_regex.search(content[:300]):
                letters.append(current_letter_pages)
                current_letter_pages = []
        
        current_letter_pages.append(page)
        
        # Check if this page ends a letter
        # We look at the last 500 chars for a closing marker
        if closing_regex.search(content[-500:]):
            letters.append(current_letter_pages)
            current_letter_pages = []
            
    if current_letter_pages:
        letters.append(current_letter_pages)
        
    return letters

def save_letters(letters, base_filename, output_dir):
    """Saves each group of pages as a separate text file."""
    for idx, letter_pages in enumerate(letters):
        output_path = os.path.join(output_dir, f"{base_filename}_{idx+1:03d}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            if letter_pages:
                f.write(f"Initial Page: {letter_pages[0]['num']}\n\n")
            for page in letter_pages:
                f.write(page['content'])
                f.write("\n\n")

def main():
    input_dir = "02_vlmOutput"
    output_dir = "03_splitDoc"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing: {filename}")
            
            pages = parse_pages(file_path)
            if not pages:
                print(f"  No pages found in {filename}")
                continue
                
            letters = split_into_letters(pages)
            print(f"  Found {len(letters)} letters.")
            
            base_name = os.path.splitext(filename)[0]
            save_letters(letters, base_name, output_dir)

if __name__ == "__main__":
    main()
