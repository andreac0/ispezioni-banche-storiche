import pandas as pd
import pathlib

folder_txt = './03_splitDoc/' 
df = pd.read_csv("./03_document_metadata.csv")

# 2. Rename original column and extract the ID prefix
df = df.rename(columns={'banca': 'banca_original'})
df['id_archivio'] = df['filename'].apply(lambda x: '_'.join(x.split('_')[:-1]))
df['id_prefix'] = df['filename'].str.split('_').str[0]

# 3. Logic to find the most common name per group
def get_most_frequent_bank(series):
    # Filter out NaN and 'UNKNOWN'
    valid_names = series.dropna()
    valid_names = valid_names[valid_names.astype(str).str.upper() != 'UNKNOWN']
    
    if not valid_names.empty:
        modes = valid_names.mode()
        if not modes.empty:
            return modes[0]
    return 'UNKNOWN'

# 4. Create the harmonized column
harmonized_map = df.groupby('id_prefix')['banca_original'].apply(get_most_frequent_bank)
df['banca'] = df['id_prefix'].map(harmonized_map)

# 5. Final cleanup: reorder columns for readability
df = df[['id_archivio','pagina_iniziale','filename', 'banca_original', 'banca', 'data', 'tipo', 'luogo', 'titolo']]

df['banca'] = df['banca'].str.upper()

def load_text_content(file_path):
    path = pathlib.Path(folder_txt+file_path)
    if path.exists():
        try:
            # Using utf-8, but you might need 'latin-1' for older Italian archives
            return path.read_text(encoding='utf-8').strip()
        except Exception as e:
            return f"Error reading file: {e}"
    return "File not found"

# Note: If your files are in a specific folder, use: 'folder_path/' + df['filename']
df['text'] = df['filename'].apply(load_text_content)

# 4. Final Organization
# Moving 'text' to the end and keeping both banca versions
cols = ['id_archivio', 'pagina_iniziale', 'filename', 'banca', 'data', 'tipo', 'luogo', 'titolo', 'text']
df = df[cols]

df.to_csv('./04_processed_data.csv', index=False)

# save rows that have text not null
df[df['text'].notna() & (df['text'] != '') & (df['text'] !='File not found')].to_csv('./with_text.csv', index=False)
