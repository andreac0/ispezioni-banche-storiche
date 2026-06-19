import pandas as pd
import re
import json
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- CONFIGURAZIONE ---
INPUT_CSV = "04_processed_data.csv"
TEXT_COLUMN = "text"
OUTPUT_CSV = "fidi_prestiti_extraction.csv"

# Model initialization
MODEL_NAME = "google/gemma-4-26B-A4B-it"
BATCH_SIZE = 96  

# --- PROMPT ---
SYSTEM_PROMPT = (
    "Sei un assistente esperto in analisi documentale storica di istituti bancari italiani (periodo 1900-1950).\n"
    "Il tuo compito è estrarre in modo strutturato informazioni su fidi, prestiti ed esposizioni creditizie della BANCA ISPEZIONATA.\n\n"
    "REGOLE DI ESTRAZIONE RIGIDE:\n"
    "1. SOGGETTO DELL'OPERAZIONE: Estrai il nome della persona fisica o dell'ente. \n"
    "   - NORMALIZZAZIONE: Usa il MAIUSCOLO. Rimuovi titoli (es. Cav., Dott., Sig., Rag., Comm.). \n"
    "   - PULIZIA: Se il nome include la località (es. 'Bozzano di Macomer'), estrai 'BOZZANO' nel nome e 'Macomer' nel campo località.\n"
    "2. IDENTIFICAZIONE: Evita soggetti collettivi generici (es. 'poveri lavoratori', 'clientela rurale') a meno che non siano l'unico riferimento per un'operazione specifica.\n"
    "3. IMPORTI E SOGLIE: Se il testo indica che un fido 'supera le 250.000 lire' senza specificare l'importo esatto, scrivi '>250000' nel campo importo_accordato.\n"
    "4. RUOLO: Per i 'Depositi/Risparmio', il soggetto è il titolare del conto (anche se tecnicamente creditore della banca).\n\n"
    "TASSONOMIA CATEGORIALE:\n"
    "- 'tipologia_fido_macro': [Sconto portafoglio, Apertura credito in C/C, Anticipazione, Mutuo/Prestito lungo termine, Fideiussione/Avallo, Sovvenzione/Prestito agrario, Deposito/Risparmio, Altro]\n"
    "- 'tipo_soggetto': [Persona fisica, Società/Ditta, Ente pubblico/Fondazione, Altro]\n"
    "- 'garanzia_tipo': [In bianco, Reale (Titoli/Merci), Reale (Ipoteca), Personale (Fideiussione/Avallo), Altro/Mista, Non specificato]\n"
    "- 'giudizio_ispettore': [regolare, dubbio, sofferenza, perdita, Non specificato]\n\n"
    "CAMPI DA ESTRARRE:\n"
    "1. 'debitore': Nome normalizzato (es. 'FRATELLI BOZZANO').\n"
    "2. 'localita': Città o sede del soggetto (se indicata).\n"
    "3. 'tipo_soggetto': Dalla tassonomia.\n"
    "4. 'tipologia_fido_dettaglio': Descrizione testuale della natura del credito.\n"
    "5. 'tipologia_fido_macro': Scegli dalla tassonomia fornita.\n"
    "6. 'importo_accordato': Valore numerico o soglia (es. '>250000') o 'Non specificato'.\n"
    "7. 'importo_utilizzato': Valore numerico o 'Non specificato'.\n"
    "8. 'garanzie_dettaglio': Descrizione delle garanzie.\n"
    "9. 'garanzia_tipo': Scegli dalla tassonomia fornita.\n"
    "10. 'tasso_interesse': Percentuale.\n"
    "11. 'scadenza': Data o periodo.\n"
    "12. 'esponente_correlato_bool': Scegli [Sì, No, Non specificato].\n"
    "13. 'esponente_correlato_note': Dettagli sul legame con la banca.\n"
    "14. 'giudizio_ispettore': Scegli dalla tassonomia fornita.\n"
    "15. 'note_ispettore': Commenti rilevanti.\n\n"
    "Restituisci ESATTAMENTE un JSON array di oggetti. Solo JSON valido. No markdown. No spiegazioni.\n"
    "Esempio: [{\"debitore\": \"FRATELLI ROSSI\", \"localita\": \"Milano\", ...}]"
)

def get_smart_chunks(text, chunk_size=10000, overlap=500):
    if not isinstance(text, str) or len(text) < 100:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    keywords = [
        "fido", "prestito", "debitore", "esposizione", "garanzia", "scoperto", 
        "mutuo", "cambiale", "sconto", "anticipazione", "ipoteca", "fideiussione", 
        "fidejussione", "castelletto", "sofferenza", "incaglio", "pendenza", "avallo", 
        "tasso", "affidamento", "sovvenzione", "portafoglio", "effetti", "titoli", 
        "pegno", "privilegio", "coobbligato", "mallevadore", "scaduto", "perito",
        "rinnovo", "estinzione", "linea di credito", "cassa", "scoperto di conto", 
        "credito agrario", "credito fondiario"
    ]

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
        data = json.loads(clean)
        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
            return [data]
        return data
    except json.JSONDecodeError:
        # tentativo di recupero per output troncato
        last_obj = clean.rfind("}")
        if last_obj != -1:
            recovered = clean[:last_obj + 1]
            if recovered.startswith("["):
                recovered += "]"
            try:
                data = json.loads(recovered)
                if isinstance(data, dict):
                    for key in data:
                        if isinstance(data[key], list):
                            return data[key]
                    return [data]
                return data
            except:
                pass
        raise

def build_record(filename, item):
    """Costruisce un record normalizzato da un item JSON del modello."""
    debitore = str(item.get('debitore', '')).strip()
    if not debitore or debitore.lower() in ['null', 'none', 'non specificato']:
        return None
    return {
        'filename': filename,
        'debitore': debitore,
        'localita': str(item.get('localita', 'Non specificato')).strip(),
        'tipo_soggetto': str(item.get('tipo_soggetto', 'Altro')).strip(),
        'tipologia_fido_dettaglio': str(item.get('tipologia_fido_dettaglio', 'Non specificato')).strip(),
        'tipologia_fido_macro': str(item.get('tipologia_fido_macro', 'Altro')).strip(),
        'importo_accordato': str(item.get('importo_accordato', 'Non specificato')).strip(),
        'importo_utilizzato': str(item.get('importo_utilizzato', 'Non specificato')).strip(),
        'garanzie_dettaglio': str(item.get('garanzie_dettaglio', 'Non specificato')).strip(),
        'garanzia_tipo': str(item.get('garanzia_tipo', 'Non specificato')).strip(),
        'tasso_interesse': str(item.get('tasso_interesse', 'Non specificato')).strip(),
        'scadenza': str(item.get('scadenza', 'Non specificato')).strip(),
        'esponente_correlato_bool': str(item.get('esponente_correlato_bool', 'No')).strip(),
        'esponente_correlato_note': str(item.get('esponente_correlato_note', 'Non specificato')).strip(),
        'giudizio_ispettore': str(item.get('giudizio_ispettore', 'Non specificato')).strip(),
        'note_ispettore': str(item.get('note_ispettore', 'Non specificato')).strip()
    }

def aggregate_results(results):
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.replace(['None', 'null', 'nan', 'Non specificato', ''], pd.NA)

    # Pulizia per aggregazione: rimuove parentesi, normalizza F.lli e rimuove patronimici
    df['debitore_agg'] = df['debitore'].str.upper()
    df['debitore_agg'] = df['debitore_agg'].str.replace(r'\bF\.?LLI\b', 'FRATELLI', regex=True)
    df['debitore_agg'] = df['debitore_agg'].str.replace(r'\b(FU|DI)\b.*$', '', regex=True)
    df['debitore_agg'] = df['debitore_agg'].str.replace(r'\(.*\)', '', regex=True).str.strip()

    def merge_logic(series):
        valid = series.dropna().astype(str)
        if valid.empty:
            return "Non specificato"
        # Ritorna la stringa più lunga (spesso la più informativa)
        return sorted(list(set(valid)), key=len, reverse=True)[0]

    final_rows = []
    # Raggruppiamo per file, debitore pulito e macro tipologia
    for (fname, debitore_clean, tipo_macro), group in df.groupby(['filename', 'debitore_agg', 'tipologia_fido_macro']):
        merged = {
            'filename': fname, 
            'debitore': debitore_clean,
            'tipologia_fido_macro': tipo_macro
        }
        for col in group.columns:
            if col not in ['filename', 'debitore', 'tipologia_fido_macro', 'debitore_agg']:
                merged[col] = merge_logic(group[col])
        final_rows.append(merged)

    return pd.DataFrame(final_rows)

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

    print(f"Totale frammenti da analizzare: {len(all_tasks)}")

    all_records = []

    # Batch inference con vLLM
    for i in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="Extracting fidi and loans"):
        batch = all_tasks[i : i + BATCH_SIZE]

        batch_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analizza questo frammento del documento '{fname}' ed estrai i fidi/prestiti:\n\n{chunk}"}
            ]
            for fname, chunk in batch
        ]

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)

            for (fname, _), output in zip(batch, outputs):
                raw_response = output.outputs[0].text.strip()
                try:
                    items = parse_json_response(raw_response)
                    if items:
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
            print(output_df.head(20))
        else:
            print("\nNessun dato dopo l'aggregazione.")
    else:
        print("\nNessun dato estratto.")

if __name__ == "__main__":
    main()
