import pandas as pd
import re
import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- CONFIGURAZIONE ---
INPUT_CSV = "04_processed_data.csv"
TEXT_COLUMN = "text"
OUTPUT_CSV = "board_members_extraction.csv"

# Model initialization
MODEL_NAME = "google/gemma-4-26B-A4B-it"
BATCH_SIZE = 96  

# --- PROMPT ---
SYSTEM_PROMPT = (
    "Sei un assistente esperto in analisi documentale storica di istituti bancari italiani (periodo 1900-1950).\n"
    "Il tuo compito è estrarre in modo strutturato i membri degli organi sociali (CdA, Collegio Sindacale) e l'Alta Dirigenza della BANCA ISPEZIONATA.\n\n"
    "REGOLE DI ESCLUSIONE TASSATIVE (PENALITÀ MASSIMA SE VIOLATE):\n"
    "1. NON ESTRARRE CLIENTI: Ignora chiunque sia citato solo come 'correntista', 'titolare di libretto', 'depositante', 'cliente', 'debitore', 'avallista' o 'fido'. In particolare, ignora i Presidenti o Direttori di ALTRE società (Cooperative, Industrie, Enti) citati solo perché la loro azienda è cliente o debitrice della banca.\n"
    "2. DISTINGUI GLI ENTI: NON ESTRARRE il personale della Banca d'Italia (Governatori, Ispettori, Capi Ufficio) che redige il rapporto.\n"
    "3. SOLO CARICHE SOCIALI NELLA BANCA ISPEZIONATA: Estrai solo chi ha un ruolo esplicito nell'organigramma della banca ispezionata. Se un soggetto è 'Presidente' di un'altra ditta o cooperativa ma non ha cariche nella banca stessa, NON ESTRARLO.\n"
    "4. AZIONI: Estrai solo azioni della BANCA ISPEZIONATA. Ignora titoli di altre società (es. Breda, Pirelli) citati come garanzia.\n\n"
    "CAMPI DA ESTRARRE:\n"
    "1. 'cognome': Cognome del soggetto.\n"
    "2. 'nome': Nome del soggetto. Se non presente, usa 'Non specificato'.\n"
    "3. 'onorificenza': Titoli onorifici ESPANSI (es. 'Cavaliere', 'Commendatore').\n"
    "4. 'professione': Titoli professionali ESPANSI (es. 'Avvocato', 'Ingegnere').\n"
    "5. 'ruolo': Ruolo nell'istituto ispezionato. Valori: [Presidente, Vice Presidente, Consigliere Delegato, Consigliere, Sindaco Effettivo, Sindaco Supplente, Direttore Generale, Segretario, Direttore di Filiale, Non specificato].\n"
    "6. 'organo': Organo di appartenenza. Valori: [Consiglio di Amministrazione, Collegio Sindacale, Alta Direzione, Non specificato].\n"
    "7. 'profilo_sociale': Descrizioni biografiche.\n"
    "8. 'parentela': Legami familiari.\n"
    "9. 'altre_cariche': Cariche in ALTRE società.\n"
    "10. 'azioni': Numero di azioni della BANCA ISPEZIONATA.\n"
    "11. 'affiliazione_politica': [PNF, Regime Fascista, Antifascista, Partigiano, CLN, Non specificato].\n"
    "12. 'dettagli_politici': Giustificazione affiliazione.\n"
    "13. 'data_luogo_nascita': Info anagrafiche.\n"
    "14. 'giudizio_ispettore': ['positivo', 'negativo', 'neutro', 'Non specificato'].\n\n"
    "Restituisci ESATTAMENTE un JSON array di oggetti. Solo JSON valido. No markdown. No spiegazioni.\n"
    "Esempio: [{\"cognome\": \"Rossi\", \"nome\": \"Mario\", ...}]"
)

# --- MAPPATURA NORMALIZZAZIONE ---
NORMALIZATION_MAP = {
    "avv": "Avvocato", "ing": "Ingegnere", "dott": "Dottore",
    "rag": "Ragioniere", "geom": "Geometra", "prof": "Professore",
    "cav": "Cavaliere", "comm": "Commendatore",
    "gr uff": "Grand'Ufficiale", "gr. uff": "Grand'Ufficiale",
    "cav lav": "Cavaliere del Lavoro", "cav. lav": "Cavaliere del Lavoro",
    "mons": "Monsignore", "on": "Onorevole", "sen": "Senatore",
}

def normalize_title(text):
    if not isinstance(text, str) or text.lower() in ['nan', 'none', 'non specificato', '']:
        return "Non specificato"
    clean_text = text.lower().replace('.', '').strip()
    return NORMALIZATION_MAP.get(clean_text, text.strip().capitalize())

def get_smart_chunks(text, chunk_size=12000, overlap=2000):
    if not isinstance(text, str) or len(text) < 100:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    keywords = ["consiglio", "amministrazione", "cda", "presidente", "direttore",
                "sindaco", "collegio", "azioni", "nomina", "soci", "esponenti"]

    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_period = text.rfind('\n', start + chunk_size // 2, end)
            if last_period != -1:
                end = last_period + 1

        chunk = text[start:end]
        if any(kw in chunk.lower() for kw in keywords):
            chunks.append(chunk)

        start = end - overlap
        if end >= len(text):
            break
    return chunks

def parse_json_response(raw_response):
    clean = raw_response.strip()

    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # tentativo di recupero per output troncato
        last_obj = clean.rfind("}")
        if last_obj != -1:
            recovered = clean[:last_obj + 1]

            if recovered.startswith("["):
                recovered += "]"

            return json.loads(recovered)

        raise

def build_record(filename, item):
    """Costruisce un record normalizzato da un item JSON del modello."""
    cognome = str(item.get('cognome', '')).strip()
    if not cognome or cognome.lower() in ['null', 'none', 'non specificato']:
        return None
    return {
        'filename': filename,
        'cognome': cognome,
        'nome': str(item.get('nome', '')).strip(),
        'onorificenza': normalize_title(str(item.get('onorificenza', ''))),
        'professione': normalize_title(str(item.get('professione', ''))),
        'ruolo': str(item.get('ruolo', '')).strip(),
        'organo': str(item.get('organo', '')).strip(),
        'profilo_sociale': str(item.get('profilo_sociale', '')).strip(),
        'parentela': str(item.get('parentela', '')).strip(),
        'altre_cariche': str(item.get('altre_cariche', '')).strip(),
        'azioni': str(item.get('azioni', '')).strip(),
        'affiliazione_politica': str(item.get('affiliazione_politica', '')).strip(),
        'dettagli_politici': str(item.get('dettagli_politici', '')).strip(),
        'data_luogo_nascita': str(item.get('data_luogo_nascita', '')).strip(),
        'giudizio_ispettore': str(item.get('giudizio_ispettore', '')).strip(),
    }

def aggregate_results(results):
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.replace(['None', 'null', 'nan', 'Non specificato', ''], pd.NA)

    def merge_logic(series):
        valid = series.dropna().astype(str)
        if valid.empty:
            return ""
        if series.name == 'azioni':
            nums = valid.str.extract('(\d+)').dropna()
            if not nums.empty:
                return nums.iloc[0, 0]
        return sorted(list(set(valid)), key=len, reverse=True)[0]

    final_rows = []
    for (fname, cognome), group in df.groupby(['filename', 'cognome']):
        unique_names = group['nome'].dropna().unique()
        if len(unique_names) <= 1:
            merged = {'filename': fname, 'cognome': cognome.strip().capitalize()}
            for col in group.columns:
                if col not in ['filename', 'cognome']:
                    merged[col] = merge_logic(group[col])
            if not merged.get('nome'):
                merged['nome'] = "Non specificato"
            final_rows.append(merged)
        else:
            for name in unique_names:
                subgroup = group[group['nome'] == name]
                merged = {'filename': fname, 'cognome': cognome.strip().capitalize(), 'nome': name.strip().capitalize()}
                for col in group.columns:
                    if col not in ['filename', 'cognome', 'nome']:
                        merged[col] = merge_logic(subgroup[col])
                final_rows.append(merged)

    if not final_rows:
        return pd.DataFrame()

    res_df = pd.DataFrame(final_rows)

    res_df = res_df[~(
        (res_df['ruolo'].isin(['', 'Non specificato', pd.NA])) &
        (res_df['organo'].isin(['', 'Non specificato', pd.NA]))
    )]

    customer_keywords = ['cliente', 'correntista', 'debitore', 'titolare',
                         'libretto', 'avallista', 'fido', 'conto corrente']

    def is_customer(row):
        text_to_check = f"{row.get('ruolo', '')} {row.get('profilo_sociale', '')} {row.get('altre_cariche', '')}".lower()
        return any(kw in text_to_check for kw in customer_keywords)

    res_df = res_df[~res_df.apply(is_customer, axis=1)]
    return res_df

def main():
    print(f"Initializing vLLM with model: {MODEL_NAME}")

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.97,
        max_model_len=32768,
        enable_prefix_caching=True,   
        enable_chunked_prefill=True,
        max_num_batched_tokens=65536,
        max_num_seqs=192,
    )

    sampling_params = SamplingParams(
        temperature=0.0,  
        max_tokens=6000,   
    )

    # Caricamento e filtro dati
    try:
        df_full = pd.read_csv(INPUT_CSV)
        df = df_full[df_full['tipo'].isin(['ispezione_rapporto', 'ispezione_prospetto'])].copy()
        print(f"File caricato: {len(df_full)} righe totali. Rapporti ispettivi: {len(df)} righe.")
    except Exception as e:
        print(f"Errore caricamento CSV: {e}")
        return

    if df.empty:
        print("Nessun documento da processare.")
        return

    # Costruzione tasks (filename, chunk)
    all_tasks = []
    for _, row in df.iterrows():
        filename = row.get('filename', 'unknown')
        text = str(row[TEXT_COLUMN])
        for chunk in get_smart_chunks(text):
            all_tasks.append((filename, chunk))

    if not all_tasks:
        print("Nessun frammento rilevante generato.")
        return

    print(f"Totale frammenti da classificare: {len(all_tasks)}")

    all_records = []

    # Batch inference con vLLM (identico al pattern del classificatore)
    for i in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="Extracting board members"):
        batch = all_tasks[i : i + BATCH_SIZE]

        batch_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analizza questo frammento del documento '{fname}':\n\n{chunk}"}
            ]
            for fname, chunk in batch
        ]

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)

            for (fname, _), output in zip(batch, outputs):
                raw_response = output.outputs[0].text.strip()
                try:
                    items = parse_json_response(raw_response)
                    for item in items:
                        record = build_record(fname, item)
                        if record:
                            all_records.append(record)
                except Exception as e:
                    print(f"JSON parse error [{fname}]: {e} | raw: {raw_response[:120]}")

        except Exception as e:
            print(f"Critical error in batch starting at index {i}: {e}")

    # Aggregazione e salvataggio
    if all_records:
        output_df = aggregate_results(all_records)
        if not output_df.empty:
            output_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"\nCompletato! Salvati {len(output_df)} record in {OUTPUT_CSV}")
            print(output_df[['cognome', 'nome', 'ruolo', 'organo', 'profilo_sociale']].head(20))
        else:
            print("\nNessun dato dopo l'aggregazione e i filtri.")
    else:
        print("\nNessun dato estratto.")

if __name__ == "__main__":
    main()