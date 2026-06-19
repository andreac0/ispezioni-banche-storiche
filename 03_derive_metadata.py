import os
import json
import csv
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- Configuration ---
INPUT_DIR = "03_splitDoc"
METADATA_CSV = "document_metadata.csv"

# Model initialization 
MODEL_NAME = "google/gemma-4-26B-A4B-it"
# google/gemma-4-31B-it-qat-w4a16-ct
GPU_UTILIZATION = 0.92          # era 0.90 — più headroom disponibile
BATCH_SIZE      = 96            # 128+ peggiora su PCIe per l'I/O CPU-side
MAX_MODEL_LEN   = 32768   

def main():

    print(f"Initializing vLLM with model: {MODEL_NAME}")
    # Initialize the LLM
    # llm = LLM(
    #     model=MODEL_NAME,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=GPU_UTILIZATION,
    #     max_model_len=MAX_MODEL_LEN,
    #     dtype="bfloat16",           # niente fp8: 80GB ne fa a meno
    #     # quantization rimossa
    #     enable_prefix_caching=True, # win enorme: system prompt ~800 tok fisso
    #     max_num_batched_tokens=8192, # limita il prefill per batch su PCIe
    #     max_num_seqs=BATCH_SIZE,
    #     # su PCIe conviene limitare i worker CPU per non saturare il bus
    #     # tensor_parallel_size=1 (default, 1 sola GPU)
    # )

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
        temperature=0.0,   # greedy: JSON deterministico + decode più veloce
        max_tokens=150,    # era 512 — l'output JSON è ~80 token
    )

    # Get list of files to process
    txt_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")])
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}. Have you run 02_split.py?")
        return

    print(f"Found {len(txt_files)} files to classify.")

    # System prompt for classification and extraction
    system_prompt = """
Sei un esperto archivista storico specializzato nell'analisi di documenti della Vigilanza bancaria italiana (1900-1970).
Il tuo compito è estrarre metadati precisi in formato JSON.

REGOLE PER I CAMPI:
1. "banca": Identifica l'istituto finanziario privato SOGGETTO del documento (es. Credito Italiano, Banco di Roma).
   - IGNORA SEMPRE: "Banca d'Italia", "Bankitalia", "Vigilanza", "Ispettorato", "Ministero", "Governatore", "Amministrazione Centrale". 
   - ATTENZIONE ERRORI OCR: Ignora varianti come "Bancard", "Bancit", "Bancit-Vigilanza". Sono errori per Banca d'Italia.
   - CORREZIONE GEOGRAFICA: Se leggi nomi di città storpiati (es. "Bimla"), correggili se il contesto è chiaro (es. "Biella").
   - Se il documento è un fido a una ditta, la banca è l'istituto che concede il credito.
   - Usa nomi completi normalizzati (es. "Banca Commerciale Ligure" invece di "Banca Commerciale Liguri").

2. "data": Formato ISO YYYY-MM-DD o YYYY. 
   - ERA FASCISTA: Se trovi "Anno XV", "Anno X", ecc., converti in anno solare (1922 + numero romano). Esempio: "1932-X" diventa "1932".
   - Se l'OCR suggerisce date palesemente errate (es. 1990 in un contesto anni '30), correggi in base al contenuto storico.

3. "tipo": Scegli ESATTAMENTE uno tra: 
   - "lettera" (corrispondenza, telegrammi, istanze, comunicazioni, avvisi, richieste, domande)
   - "ispezione_incarico" (disposizione di avvio ispezione)
   - "ispezione_rapporto" (SOLO rapporti ispettivi)
   - "bilancio" (situazioni contabili, patrimoniali, conti profitti/perdite, elenchi fidi)
   - "statuto" (statuti, modifiche statutarie)
   - "verbale_assemblea" (verbali di assemblee soci o CdA)
   - "altro"

4. "luogo": La città della FILIALE o della sede della banca privata oggetto del documento.
   - NON usare "Roma" se è solo l'indirizzo del destinatario (Banca d'Italia). 
   - Esempio: Se è un rapporto sulla "filiale di Faenza" indirizzato a Roma, il luogo è "Faenza".

5. "titolo": Titolo professionale in italiano che riassuma l'oggetto del documento.

Restituisci ESATTAMENTE questo JSON:
{
  "banca": "string",
  "data": "string",
  "tipo": "string",
  "luogo": "string",
  "titolo": "string"
}
Solo JSON valido. No markdown. No spiegazioni.
"""

    all_metadata = []

    # Process in batches for high throughput with VLLM
    for i in tqdm(range(0, len(txt_files), BATCH_SIZE), desc="Classifying documents"):
        batch_files = txt_files[i : i + BATCH_SIZE]
        batch_messages = []
        batch_page_nums = []

        for filename in batch_files:
            file_path = os.path.join(INPUT_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(8000) 
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                content = ""

            # Deterministically extract the initial page number
            page_num_match = re.search(r"^Initial Page:\s*(\d+)", content)
            page_num = page_num_match.group(1) if page_num_match else "Unknown"
            batch_page_nums.append(page_num)

            # Prepare the message for the chat API
            batch_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Testo del documento:\n\n{content}"}
            ])

        # Batch inference
        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params, use_tqdm=False)
            
            for filename, output, page_num in zip(batch_files, outputs, batch_page_nums):
                raw_response = output.outputs[0].text.strip()
                
                # Attempt to parse the JSON response
                try:
                    # Clean up common LLM artifacts like markdown blocks
                    clean_json = raw_response
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()
                    
                    metadata = json.loads(clean_json)
                except Exception as e:
                    # Fallback if JSON parsing fails
                    metadata = {
                        "banca": "Unknown", 
                        "data": "Unknown", 
                        "tipo": "altro",
                        "luogo": "Unknown",
                        "titolo": "Unknown",
                        "parsing_error": str(e),
                        "raw_output_preview": raw_response[:100]
                    }
                
                metadata['filename'] = filename
                metadata['pagina_iniziale'] = page_num
                all_metadata.append(metadata)

        
        except Exception as e:
            print(f"Critical error in batch starting at index {i}: {e}")

    # Generate summary CSV report
    if all_metadata:
        with open(METADATA_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'pagina_iniziale', 'banca', 'data', 'tipo','luogo','titolo'])
            writer.writeheader()
            for m in all_metadata:
                writer.writerow({
                    'filename': m.get('filename', ''),
                    'pagina_iniziale': m.get('pagina_iniziale', 'Sconosciuto'),
                    'banca': m.get('banca', 'Sconosciuto'),
                    'data': m.get('data', 'Sconosciuto'),
                    'tipo': m.get('tipo', 'altro'),
                    'luogo': m.get('luogo','Sconosciuto'),
                    'titolo': m.get('titolo','Sconosciuto')
                })

    print(f"\nClassification complete.")
    print(f"- Summary report saved to: {METADATA_CSV}")

if __name__ == "__main__":
    main()
