# Bildempfehlungssystem – Big Data Projekt

## 1. Ziel des Projekts
Dieses Projekt ist im Rahmen des Kurses **„Big Data“** entstanden.  
Es handelt sich um ein **Bildempfehlungssystem**, das ein Eingabebild mit einem großen Bilddatensatz anhand mehrerer Ähnlichkeitsmetriken vergleicht:

- **Farbe** (Farb-Histogramme)  
- **Hashing** (Perceptual Hash)  
- **Embeddings** (z. B. Vektor-Repräsentationen aus Modellen)  

Zur **Verbesserung der Skalierbarkeit** werden schnelle Approximate Nearest Neighbor (**ANN**)-Suchverfahren eingesetzt.

---

## 2. Programmdesign & Architektur
Die Architektur besteht aus drei Schritten:

### 2.1 Speichern
- Ausführen von `setup_db.py`  
- Das Skript verarbeitet alle Bilder im angegebenen Ordner und berechnet:  
  - Farb-Histogramme  
  - Embeddings  
  - Perceptual Hashes  
- Speicherung in **SQLite-Datenbank `images.db`**  
  - Features werden als Pickle-BLOBs abgelegt (z. B. NumPy-Vektoren)  
  - Metadaten und Dateipfade werden ebenfalls gespeichert  
- Zusätzlich wird eine Debug-Kopie `image_metrics.pkl` erstellt (lesbares Format: Listen, Strings, Zahlen).  

### 2.2 Indizieren
- Ausführen von `build_indexes.py`  
- Die gespeicherten Merkmale werden aus der Datenbank geladen und in **FAISS-Indizes** überführt:  
  - **Embeddings** → IVF-PQ Index  
  - **Farb-Histogramme** → HNSW Index  
- Indizes werden als `.faiss`-Dateien gespeichert.  
- Vorteil: schnelle ANN-Suche im Gegensatz zur „rohen“ Datenbank.  

### 2.3 Suchen
- Ausführen von `main.py` mit Pfad zu einem Eingabebild.  
- Schritte:  
  1. Features für das neue Bild berechnen  
  2. Suche in FAISS-Indizes nach ähnlichen Kandidaten  
  3. Zusätzliche Infos bei Bedarf aus der Datenbank holen  
  4. Ergebnisse der drei Vergleiche (Embedding, Farbe, Hash) zusammenführen  
  5. Sortierte Ergebnisliste zurückgeben  

⚡ **Wichtig:**  
- `setup_db.py` und `build_indexes.py` müssen nur **einmal** ausgeführt werden (Initialisierung).  
- Danach reicht es, mit `main.py` zu arbeiten.

---

## 3. Klassendesign
Alle drei Metriken folgen dem gleichen Muster:

- **ColorSimilarity**  
- **HashingSimilarity**  
- **EmbeddingSimilarity**

Jede Klasse umfasst drei Kernbereiche:

1. `compute_feature()`  
   - Berechnet Feature-Vektor bzw. Hash aus einem Bild  

2. **Index-Handling**  
   - Bauen / Laden von FAISS-Indizes (bei Farbe & Embeddings)  

3. `find_similar()`  
   - Suche nach ähnlichen Bildern  
   - Zwei Modi:  
     - **ANN-Search** über FAISS (schnell)  
     - **Fallback:** Vollständiger Scan der Datenbank (falls kein Index vorhanden)  

---

## 4. Installation & Nutzung

### Voraussetzungen
- Python 3.8+  
- Abhängigkeiten (z. B. in `requirements.txt`):  
  - `faiss`  
  - `numpy`  
  - `opencv-python`  
  - `sqlite3` (Standardbibliothek)  
  - `pickle` (Standardbibliothek)  

### Setup
```bash
# 1. Datenbank aufbauen
python setup_db.py --input <Bilderordner>

# 2. Indizes erstellen
python build_indexes.py
```

### Suche starten
```bash
python main.py --query <pfad/zum/bild.jpg>
```

---

## 5. Ergebnis
- Ausgabe: Liste ähnlicher Bilder, sortiert nach Relevanz  
- Kombination der drei Metriken für robustere Ergebnisse  
