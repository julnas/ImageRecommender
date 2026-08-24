# Bildempfehlungssystem – Big-Data-Projekt

## 1. Ziel des Projekts

Dieses Projekt ist ein Bildempfehlungssystem, das ein Eingabebild anhand von
drei Ähnlichkeitsmetriken mit einer großen Bildsammlung vergleicht:

- Farbhistogramme
- visuelle Embeddings
- Average Hash (aHash)

Für die Farb- und Embeddingsuche werden FAISS-Indizes für schnelle
Approximate-Nearest-Neighbor-Suchen (ANN) verwendet. Die kompakten Hashwerte
werden direkt über ihre Hamming-Distanz verglichen.

## 2. Architektur

Das Projekt besteht aus einer Offline-Vorbereitung und einer Online-Suche.

### 2.1 Datenbank aufbauen

`setup_db.py` durchsucht den konfigurierten Bildordner rekursiv. Für jedes
gültige Bild werden Farbhistogramm, Embedding, aHash und Metadaten berechnet.
Die Ergebnisse werden in `images_database.db` gespeichert. Farbhistogramme und
Embeddings liegen als Pickle-BLOBs vor, während der Hash als kompakter
8-Byte-BLOB gespeichert wird. Dateipfade werden relativ zum ausgewählten
Bildordner abgelegt.

Der Aufbau erfolgt zunächst in einer temporären Datenbank. Die bestehende
Datenbank wird erst ersetzt, wenn der vollständige Scan erfolgreich beendet
wurde. Ein fehlgeschlagener Scan lässt den vorherigen Stand unverändert.

### 2.2 FAISS-Indizes aufbauen

`build_indexes.py` liest die gespeicherten Merkmale aus der Datenbank und
erzeugt zwei Suchindizes:

- Embeddings: IVF-PQ
- Farbhistogramme: HNSW

Die Dateien werden im Ordner `indexes/` gespeichert. Auch hier werden neue
Indizes zunächst in einem temporären Ordner erstellt und erst nach erfolgreicher
Prüfung installiert.

### 2.3 Ähnliche Bilder suchen

`main.py` berechnet die Merkmale eines oder mehrerer Eingabebilder und führt
die Suche für alle aktivierten Metriken aus. Farbe und Embeddings verwenden die
FAISS-Indizes. Die Hashsuche arbeitet mit den in den Arbeitsspeicher geladenen
8-Byte-Hashes. Das Programm gibt für jede Metrik eine eigene Top-k-Liste aus
und stellt die Referenz- und Ergebnisbilder grafisch dar.

## 3. Installation

Empfohlen wird Python 3.11 mit einer virtuellen Umgebung:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --python .venv/bin/python -r requirements.txt
```

Für Tests und Formatierungsprüfungen werden zusätzlich die
Entwicklungsabhängigkeiten benötigt:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

## 4. Ausführung

### 4.1 Offline-Vorbereitung

1. In `setup_db.py` am Ende der Datei `base_dir` auf den Bildordner setzen.
2. Falls nötig `db_path`, `max_images` oder `commit_batch_size` anpassen.
3. Datenbank aufbauen:

```bash
.venv/bin/python setup_db.py
```

4. Danach die FAISS-Indizes aus der fertigen Datenbank erzeugen:

```bash
.venv/bin/python build_indexes.py
```

Die Offline-Vorbereitung muss erneut ausgeführt werden, wenn eine andere
Bildsammlung verwendet oder der bestehende Datenbestand verändert wird.

### 4.2 Online-Suche

In `main.py` im Konfigurationsbereich folgende Werte anpassen:

- `db_path`: Pfad zur Datenbank
- `base_dir`: Basisordner der gespeicherten Bilder
- `comparing_image_path`: einzelner Bildpfad oder Liste von Bildpfaden
- `best_k`: Anzahl der Treffer pro Metrik
- `metrics`: aktivierte Ähnlichkeitsmetriken

Anschließend die Suche starten:

```bash
.venv/bin/python main.py
```

## 5. Qualitätssicherung

Formatierung prüfen:

```bash
.venv/bin/black --check .
```

Tests ausführen:

```bash
.venv/bin/python -m pytest tests -q
```

Die GitHub-Actions-Pipeline führt beide Prüfungen automatisch für Pushes und
Pull Requests auf `main` aus.
