# Rapporti Ispettivi - Data Extraction & Analysis

Questo progetto si occupa dell'estrazione, classificazione e analisi di documenti storici della Vigilanza Bancaria Italiana (1900-1970). Il workflow trasforma output OCR grezzi in dati strutturati per analisi storiche ed economiche.

## Workflow dei Dati

Il processo di elaborazione è suddiviso in diverse fasi, ognuna delle quali genera o integra file CSV:

1.  **OCR con vLLM (`01_ocr_vllm.py`):** Scansiona le cartelle di immagini JPEG (es. da archivi digitalizzati) e utilizza il modello olmOCR-2-7B tramite vLLM per estrarre il testo. Le immagini vengono ridimensionate e processate in batch per ottimizzare l'uso della GPU (A100 80GB). I file di output sono nomi flat delle cartelle (es. `8720_0_1_0_0.txt`).
    *   **Input:** Cartelle con immagini `.jpg`/`.jpeg` in `/home/azureuser/export_asbi`
    *   **Output:** File `.txt` nella cartella `vlmOutput/`

2.  **Splitting Documenti (`02_split.py`):** Legge i file OCR generati dalla fase precedente, li suddivide in singoli documenti/lettere utilizzando euristiche (chiusure formali come "distinti ossequi", marcatori di inizio come "OGGETTO"). Filtra le pagine con testo ripetutivo (possibili allucinazioni del modello).
    *   **Input:** File `.txt` in `02_vlmOutput/`
    *   **Output:** Frammenti di testo singoli in `03_splitDoc/` (es. `8720_0_1_0_0_001.txt`)

3.  **Classificazione e Metadati (`03_derive_metadata.py`):** Analizza i frammenti di testo estratti (da `03_splitDoc/`) e utilizza modelli LLM (Gemma-4) per identificare la banca, la data, il tipo di documento, il luogo e un titolo sintetico.
    *   **Output:** `03_document_metadata.csv`

4.  **Integrazione Testuale (`04_clean_and_integrate.py`):** Unisce i metadati estratti con il contenuto testuale originale dei file `.txt`.
    *   **Output:** `04_processed_data.csv`, che funge da dataset principale per le fasi successive.

5.  **Estrazione Organi Sociali (`05_CDAtests.py`):** Analizza i rapporti d'ispezione per estrarre informazioni sui membri del Consiglio di Amministrazione, del Collegio Sindacale e dell'Alta Dirigenza.
    *   **Output:** `05_board_members_extraction.csv`

6.  **Estrazione Fidi e Prestiti (`06_FidiExtraction.py`):** Estrae dettagli sulle esposizioni creditizie, i debitori, le garanzie e i giudizi degli ispettori.
    *   **Output:** `fidi_prestiti_extraction.csv`

## Relazione tra i CSV

Tutti i file sono relazionati tramite la colonna `filename`, che identifica univocamente il frammento di testo analizzato.

*   `04_processed_data.csv`: È il "cuore" del progetto. Contiene il testo e i metadati di base.
*   `05_board_members_extraction.csv`: Ogni riga rappresenta una persona fisica identificata in un documento specifico (`filename`).
*   `fidi_prestiti_extraction.csv`: Ogni riga rappresenta un'operazione di fido o un debitore citato in un documento specifico (`filename`).

---

## Schema del Database Relazionale

Per una gestione efficiente e interrogazioni complesse (es. "Tutti i fidi concessi da banche il cui CdA include il Commendatore X"), si propone il seguente schema relazionale.

### 1. Tabella `documenti`
Contiene l'anagrafica dei frammenti documentali. Estratta da `03_derive_metadata.py` tramite LLM (Gemma-4).

*   `filename` (**PK**): Testo (es. `10135_0_3_0_0_020.txt`).
*   `id_archivio`: Testo (identificativo del faldone/unità d'archivio).
*   `pagina_iniziale`: Intero (pagina nel documento originale, estratta dall'intestazione dello split).
*   `banca`: Testo — Nome normalizzato dell'istituto finanziario **privato** soggetto del documento. Il modello esclude sistematicamente "Banca d'Italia", "Bankitalia", "Vigilanza", "Ispettorato" e gestisce errori OCR (es. "Bancard" → Banca d'Italia). Se il documento è un fido, la banca è il concedente il credito.
*   `data`: Data — Formato ISO YYYY-MM-DD o YYYY. Le date dell'era fascista ("Anno XV") vengono convertite in anno solare (1922 + numero romano). Date OCR palesemente errate vengono corrette in base al contesto storico.
*   `tipo`: **Enum** — Classificazione semantica del documento:
    *   `lettera`: corrispondenza, telegrammi, istanze, comunicazioni, avvisi, richieste, domande
    *   `ispezione_incarico`: disposizione di avvio ispezione
    *   `ispezione_rapporto`: rapporti ispettivi (solo)
    *   `bilancio`: situazioni contabili, patrimoniali, conti profitti/perdite, elenchi fidi
    *   `statuto`: statuti, modifiche statutarie
    *   `verbale_assemblea`: verbali di assemblee soci o CdA
    *   `altro`
*   `luogo`: Testo — Città della **filiale o sede della banca privata** oggetto del documento (non l'indirizzo del destinatario Banca d'Italia).
*   `titolo`: Testo — Sintesi professionale dell'oggetto del documento.
*   `testo_ocr`: Text (contenuto integrale).

### 2. Tabella `organi_sociali`
Informazioni sulle persone fisiche con cariche nella banca ispezionata. Estratta da `05_CDA.py` tramite LLM (Gemma-4). Il prompt applica regole di esclusione rigide: vengono scartati clienti, correntisti, debitori, personale della Banca d'Italia e presidenti/direttori di altre società citati solo come debitori.

*   `id` (**PK**): Auto-increment.
*   `filename` (**FK** -> `documenti.filename`).
*   `cognome`: Testo — Cognome normalizzato ( maiuscolo, title-case dopo aggregazione).
*   `nome`: Testo — Nome del soggetto. "Non specificato" se assente nel documento.
*   `onorificenza`: Testo — Titoli onorifici espansi (es. "Comm" → "Commendatore", "Gr Uff" → "Grand'Ufficiale", "Cav Lav" → "Cavaliere del Lavoro").
*   `professione`: Testo — Titoli professionali espansi (es. "Avv" → "Avvocato", "Ing" → "Ingegnere", "Rag" → "Ragioniere").
*   `ruolo`: **Enum** — Ruolo esplicito nell'organigramma della banca ispezionata:
    *   `Presidente`, `Vice Presidente`, `Consigliere Delegato`, `Consigliere`
    *   `Sindaco Effettivo`, `Sindaco Supplente` (Collegio Sindacale)
    *   `Direttore Generale`, `Segretario`, `Direttore di Filiale`
    *   `Non specificato`
*   `organo`: **Enum** — Organo di appartenenza:
    *   `Consiglio di Amministrazione`
    *   `Collegio Sindacale`
    *   `Alta Direzione` (Direttore Generale, Segretario)
    *   `Non specificato`
*   `profilo_sociale`: Text — Note biografiche, cariche precesse, ruoli pubblici.
*   `parentela`: Testo — Legami familiari con altri esponenti della banca o del sistema economico.
*   `altre_cariche`: Text — Ruoli in **altre** società (cooperative, industrie, enti). Attenzione: se il soggetto è Presidente di un'altra ditta ma non ha cariche nella banca ispezionata, NON viene estratto.
*   `azioni`: Testo/Intero — Numero di azioni possedute nella **banca ispezionata** (non titoli di altre società usati come garanzia).
*   `affiliazione_politica`: **Enum** — Appartenenza politica nell'ambito del regime fascista:
    *   `PNF` (Partito Nazionale Fascista)
    *   `Regime Fascista` (collaborazionisti generici)
    *   `Antifascista`, `Partigiano`, `CLN` (Comitato di Liberazione Nazionale)
    *   `Non specificato`
*   `dettagli_politici`: Text — Giustificazione testuale dell'affiliazione politica.
*   `data_luogo_nascita`: Text — Informazioni anagrafiche quando disponibili.
*   `giudizio_ispettore`: **Enum** — Valutazione dell'ispettore sul soggetto:
    *   `positivo`, `negativo`, `neutro`, `Non specificato`

### 3. Tabella `fidi_prestiti`
Dettagli sulle esposizioni creditizie. Estratta da `06_FidiExtraction.py` tramite LLM (Gemma-4). I nomi dei debitori vengono normalizzati in MAIUSCOLO con rimozione titoli (Dott., Sig., Rag., Comm.) e patronimici (fu, di). Soggetti collettivi generici ("poveri lavoratori", "clientela rurale") vengono scartati.

*   `id` (**PK**): Auto-increment.
*   `filename` (**FK** -> `documenti.filename`).
*   `debitore`: Testo — Nome normalizzato in maiuscolo del beneficiario del credito (es. "FRATELLI BOZZANO"). Per depositi/risparmio, è il titolare del conto.
*   `localita`: Testo — Città o sede del soggetto (se indicata; estratta dal nome quando presente, es. "Bozzano di Macomer" → localita="Macomer").
*   `tipo_soggetto`: **Enum** — Natura giuridica del debitore:
    *   `Persona fisica`
    *   `Società/Ditta`
    *   `Ente pubblico/Fondazione`
    *   `Altro`
*   `tipologia_fido_macro`: **Enum** — Categoria dell'esposizione creditizia:
    *   `Sconto portafoglio` (sconto di cambiali)
    *   `Apertura credito in C/C` (fido in conto corrente)
    *   `Anticipazione` (su merci o titoli)
    *   `Mutuo/Prestito lungo termine`
    *   `Fideiussione/Avallo` (garanzia prestata)
    *   `Sovvenzione/Prestito agrario`
    *   `Deposito/Risparmio`
    *   `Altro`
*   `tipologia_fido_dettaglio`: Testo — Descrizione testuale della natura specifica del credito.
*   `importo_accordato`: Testo — Valore numerico, soglia (es. `>250000` per "supera le 250.000 lire") o "Non specificato".
*   `importo_utilizzato`: Testo/Numerico — Importo effettivamente utilizzato.
*   `garanzia_tipo`: **Enum** — Tipologia di garanzia:
    *   `In bianco` (firma sola)
    *   `Reale (Titoli/Merci)` (pegno, portafoglio)
    *   `Reale (Ipoteca)` (immobiliare)
    *   `Personale (Fideiussione/Avallo)` (coobbligazione)
    *   `Altro/Mista`
    *   `Non specificato`
*   `garanzie_dettaglio`: Text — Descrizione analitica delle garanzie (es. "portafoglio effetti commerciali", "ipoteca su immobile in via Roma").
*   `tasso_interesse`: Testo — Percentuale o condizioni.
*   `scadenza`: Testo — Data o periodo di scadenza.
*   `esponente_correlato_bool`: **Enum** [`Sì`, `No`, `Non specificato`] — Flag che indica se il fido è concesso a un membro del CdA, Collegio Sindacale o Alta Direzione della banca ispezionata (insider lending).
*   `esponenti_correlati_note`: Text — Dettagli sul legame familiare o professionale con gli esponenti della banca.
*   `giudizio_ispettore`: **Enum** — Classificazione della qualità del credito secondo l'ispettore:
    *   `regolare` (scoperto, normale)
    *   `dubbio` (in via di deterioramento)
    *   `sofferenza` (credito in sofferenza, probabile perdita)
    *   `perdita` (credito definitivamente incagliato)
    *   `Non specificato`
*   `note_ispettore`: Text — Commenti e valutazioni qualitative dell'ispettore.

---

## Query di Esempio (SQL)

Ecco alcuni esempi di analisi che possono essere effettuate interrogando il database:

### 1. Composizione del CdA di una banca in un anno specifico
```sql
SELECT os.cognome, os.nome, os.ruolo, os.onorificenza
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE d.banca LIKE '%Popolare di Modena%'
AND d.data LIKE '1933%'
AND os.organo = 'Consiglio di Amministrazione'
ORDER BY os.cognome;
```

### 2. Analisi della qualità del credito (Sofferenze e Perdite)
```sql
SELECT d.banca, COUNT(*) as numero_fidi_critici, SUM(CAST(fp.importo_utilizzato AS FLOAT)) as totale_esposizione
FROM fidi_prestiti fp
JOIN documenti d ON fp.filename = d.filename
WHERE fp.giudizio_ispettore IN ('sofferenza', 'perdita')
GROUP BY d.banca
ORDER BY numero_fidi_critici DESC;
```

### 3. Ricerca di esponenti con affiliazione politica specifica
```sql
SELECT os.cognome, os.nome, os.affiliazione_politica, d.banca, d.luogo
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE os.affiliazione_politica = 'PNF'
AND d.tipo = 'ispezione_rapporto';
```

### 4. Fidi concessi a soggetti correlati (Insider Lending)
```sql
SELECT fp.debitore, fp.importo_accordato, fp.giudizio_ispettore, d.banca, fp.esponente_correlato_note
FROM fidi_prestiti fp
JOIN documenti d ON fp.filename = d.filename
WHERE fp.esponente_correlato_bool = 'Sì'
OR fp.esponente_correlato_note IS NOT NULL;
```

### 5. Distribuzione geografica delle ispezioni e tipologia di istituti
```sql
SELECT luogo, COUNT(*) as numero_ispezioni, GROUP_CONCAT(DISTINCT banca) as istituti_coinvolti
FROM documenti
WHERE tipo = 'ispezione_rapporto'
GROUP BY luogo
HAVING numero_ispezioni > 1;
```

### 6. Consiglieri "polivalenti": persone con cariche in più banche
```sql
SELECT os.cognome, os.nome, COUNT(DISTINCT d.banca) as num_banche,
       GROUP_CONCAT(DISTINCT d.banca) as banche
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE os.organo = 'Consiglio di Amministrazione'
GROUP BY os.cognome, os.nome
HAVING num_banche > 1
ORDER BY num_banche DESC;
```

### 7. Evoluzione del CdA nel tempo (nomine e uscite)
```sql
SELECT d.data, os.cognome, os.nome, os.ruolo, d.banca
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE d.banca LIKE '%Credito Italiano%'
AND os.organo = 'Consiglio di Amministrazione'
AND d.tipo IN ('ispezione_rapporto', 'verbale_assemblea')
ORDER BY d.data, os.cognome;
```

### 8. Qualità del credito per tipologia di garanzia
```sql
SELECT fp.garanzia_tipo,
       COUNT(*) as totale_fidi,
       SUM(CASE WHEN fp.giudizio_ispettore = 'sofferenza' THEN 1 ELSE 0 END) as sofferenze,
       SUM(CASE WHEN fp.giudizio_ispettore = 'perdita' THEN 1 ELSE 0 END) as perdite,
       ROUND(100.0 * SUM(CASE WHEN fp.giudizio_ispettore IN ('sofferenza','perdita') THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_critici
FROM fidi_prestiti fp
GROUP BY fp.garanzia_tipo
HAVING totale_fidi > 5
ORDER BY pct_critici DESC;
```

### 9. Insider lending: fidi a esponenti con dettagli familiari
```sql
SELECT fp.debitore, fp.importo_accordato, fp.tipologia_fido_macro,
       fp.giudizio_ispettore, os.cognome as esponente, os.ruolo,
       os.parentela, fp.esponenti_correlati_note
FROM fidi_prestiti fp
JOIN documenti d ON fp.filename = d.filename
JOIN organi_sociali os ON os.filename = d.filename
WHERE fp.esponente_correlato_bool = 'Sì'
AND os.organo IN ('Consiglio di Amministrazione', 'Collegio Sindacale')
ORDER BY fp.debitore;
```

### 10. Reti familiari nella dirigenza bancaria
```sql
SELECT os.parentela, os.cognome, os.nome, os.ruolo, os.organo, d.banca
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE os.parentela IS NOT NULL
AND os.parentela != ''
AND os.parentela != 'Non specificato'
ORDER BY d.banca, os.parentela;
```

### 11. Concentrazione del credito: top debitori per banca
```sql
SELECT d.banca, fp.debitore, fp.tipo_soggetto,
       COUNT(*) as num_operazioni,
       GROUP_CONCAT(DISTINCT fp.tipologia_fido_macro) as tipologie,
       MAX(fp.importo_accordato) as max_importo
FROM fidi_prestiti fp
JOIN documenti d ON fp.filename = d.filename
WHERE fp.giudizio_ispettore != 'Non specificato'
GROUP BY d.banca, fp.debitore, fp.tipo_soggetto
HAVING num_operazioni > 2
ORDER BY d.banca, num_operazioni DESC;
```

### 12. Banche con più segnali di allarme (ispezioni frequenti + fidi critici)
```sql
SELECT d.banca,
       COUNT(DISTINCT CASE WHEN d.tipo = 'ispezione_rapporto' THEN d.filename END) as rapporti_ispettivi,
       COUNT(CASE WHEN fp.giudizio_ispettore IN ('sofferenza','perdita') THEN 1 END) as fidi_critici,
       COUNT(CASE WHEN fp.esponente_correlato_bool = 'Sì' THEN 1 END) as insider_lending
FROM documenti d
LEFT JOIN fidi_prestiti fp ON fp.filename = d.filename
GROUP BY d.banca
HAVING rapporti_ispettivi > 0
ORDER BY fidi_critici DESC, insider_lending DESC;
```

### 13. Distribuzione politica degli esponenti bancari
```sql
SELECT os.affiliazione_politica,
       COUNT(*) as num_esponenti,
       COUNT(DISTINCT d.banca) as num_banche,
       GROUP_CONCAT(DISTINCT os.cognome) as nomi
FROM organi_sociali os
JOIN documenti d ON os.filename = d.filename
WHERE os.affiliazione_politica != 'Non specificato'
GROUP BY os.affiliazione_politica
ORDER BY num_esponenti DESC;
```

### 14. Timeline delle ispezioni per banca
```sql
SELECT d.banca,
       MIN(d.data) as prima_ispezione,
       MAX(d.data) as ultima_ispezione,
       COUNT(*) as totale_ispezioni,
       ROUND(JULIANDAY(MAX(d.data)) - JULIANDAY(MIN(d.data))) / 365.0 as periodo_anni
FROM documenti d
WHERE d.tipo = 'ispezione_rapporto'
GROUP BY d.banca
HAVING totale_ispezioni > 1
ORDER BY prima_ispezione;
```

### 15. Crossover tra affiliazione politica e qualità del credito
```sql
SELECT os.affiliazione_politica,
       fp.giudizio_ispettore,
       COUNT(*) as numero_fidi
FROM fidi_prestiti fp
JOIN documenti d ON fp.filename = d.filename
JOIN organi_sociali os ON os.filename = d.filename
WHERE os.affiliazione_politica != 'Non specificato'
AND fp.giudizio_ispettore != 'Non specificato'
GROUP BY os.affiliazione_politica, fp.giudizio_ispettore
ORDER BY os.affiliazione_politica, numero_fidi DESC;
```

