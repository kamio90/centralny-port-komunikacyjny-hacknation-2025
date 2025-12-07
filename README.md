# Chmura+ v2.0

**HackNation 2025 - Centralny Port Komunikacyjny**

Automatyczna klasyfikacja elementów infrastruktury na podstawie chmur punktow LAS/LAZ.

![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## Funkcjonalnosci

- **36 klas infrastruktury**: ASPRS, Railway, Road, BIM, CPK Custom
- **Wysoka wydajnosc**: ~400k punktow/s w trybie szybkim
- **Duze pliki**: Obsluga 10GB+ (tiling, memory-mapped I/O)
- **Raporty**: TXT + JSON + opcjonalny IFC
- **Interfejs**: Web UI (Streamlit) + CLI

---

## Szybki start

### Wymagania

- Python 3.9+
- RAM: 8GB minimum (16GB+ zalecane)

### Instalacja lokalna

```bash
# Sklonuj repozytorium
git clone <repo-url>
cd cpk-clasificator

# Stworz srodowisko wirtualne
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Zainstaluj zaleznosci
pip install -r requirements.txt
```

### Uruchomienie

**Web UI:**
```bash
streamlit run app.py
```
Otworz: http://localhost:8501

**CLI:**
```bash
# Podstawowe
python cli.py data/input.las output/classified.las

# Z raportem
python cli.py data/input.las output/classified.las --report output/raport.json

# Tryb szybki
python cli.py data/input.las output/classified.laz --fast
```

---

## Docker

### Build

```bash
docker build -t cpk-classifier .
```

### Uruchom Web UI

```bash
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name cpk \
  cpk-classifier
```

Otworz: http://localhost:8501

### Uruchom CLI

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  cpk-classifier \
  python cli.py /app/data/input.las /app/output/output.las --report /app/output/raport.json
```

### Zatrzymaj

```bash
docker stop cpk && docker rm cpk
```

---

## Klasy infrastruktury

### ASPRS Standard (1-9)
| ID | Nazwa |
|----|-------|
| 1 | Nieklasyfikowane |
| 2 | Grunt |
| 3 | Niska roslinnosc (<0.5m) |
| 4 | Srednia roslinnosc (0.5-2m) |
| 5 | Wysoka roslinnosc (>2m) |
| 6 | Budynki |
| 7 | Szum |
| 9 | Woda |

### Railway (17-23)
| ID | Nazwa |
|----|-------|
| 17 | Obiekt naziemny |
| 18 | Tory kolejowe |
| 19 | Linie energetyczne |
| 20 | Slupy trakcyjne |
| 21 | Perony |
| 22 | Podklady kolejowe |
| 23 | Infrastruktura kolejowa |

### Road (30-38)
| ID | Nazwa |
|----|-------|
| 30 | Jezdnia |
| 31 | Chodnik |
| 32 | Kraweznik |
| 33 | Oznakowanie poziome |
| 34 | Bariery |
| 35 | Znaki drogowe |
| 36 | Slupy oswietleniowe |
| 37 | Sygnalizacja |
| 38 | Inne elementy drogowe |

### BIM (40-47)
| ID | Nazwa |
|----|-------|
| 40 | Fundamenty |
| 41 | Sciany |
| 42 | Dachy |
| 43 | Instalacje MEP |
| 44 | Konstrukcje stalowe |
| 45 | Prefabrykaty |
| 46 | Wykonczenie |
| 47 | Tereny zielone |

### CPK Custom (64-67)
| ID | Nazwa |
|----|-------|
| 64 | Terminal lotniskowy |
| 65 | Pas startowy |
| 66 | Wezel kolejowy |
| 67 | Parking |

---

## Architektura

```
LAS/LAZ → LASLoader → TilingEngine → FeatureExtractor → Classifiers → LASWriter → LAS/LAZ + Raport
```

### Algorytmy

- **CSF** (Cloth Simulation Filter) - detekcja gruntu
- **HAG** (Height Above Ground) - strefy wysokosci
- **PCA** - cechy geometryczne (planarity, linearity, sphericity)
- **NDVI** - wykrywanie roslinnosci z RGB
- **RANSAC** - detekcja plaszczyzn

### Struktura katalogow

```
cpk-clasificator/
├── app.py              # Web UI (Streamlit)
├── cli.py              # Command Line Interface
├── src/v2/
│   ├── core/           # LASLoader, LASWriter, TilingEngine
│   ├── features/       # GeometricFeatureExtractor
│   ├── classifiers/    # 36 klasyfikatorow
│   └── pipeline/       # ClassificationPipeline
├── data/               # Pliki wejsciowe
├── output/             # Wyniki
├── Dockerfile
└── requirements.txt
```

---

## CLI - opcje

| Opcja | Opis |
|-------|------|
| `--fast, -f` | Tryb szybki (2-3x szybszy) |
| `--report, -r` | Raport JSON |
| `--report-txt` | Raport TXT |
| `--ifc` | Eksport IFC (BIM) |
| `--no-buildings` | Pomin budynki |
| `--no-infrastructure` | Pomin infrastrukture |
| `--quiet, -q` | Ciche dzialanie |
| `--verbose, -v` | Szczegolowe logi |

---

## Formaty wyjsciowe

### LAS/LAZ
Plik z nadanymi klasami w polu `classification`.

### Raport JSON
```json
{
  "metadata": {
    "input_file": "input.las",
    "processing_time_seconds": 120.5,
    "points_per_second": 400000
  },
  "statistics": {
    "total_points": 48000000,
    "classified_points": 45600000,
    "unclassified_points": 2400000
  },
  "classification": {
    "2": {"count": 20000000, "percentage": 41.67},
    "5": {"count": 10000000, "percentage": 20.83}
  }
}
```

### IFC (opcjonalnie)
Podstawowe elementy BIM wyekstrahowane z klasyfikacji.

---

## Dodawanie nowych klas

```python
from src.v2.classifiers import register_classifier, BaseClassifier

@register_classifier(class_id=100)
class MyClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(class_id=100, class_name="Moja klasa", priority=50)

    def classify(self, coords, features, height_zones, colors, intensity):
        mask = (features['planarity'] > 0.8) & (height_zones == 0)
        return mask
```

---

## Wymagania hackathonu - checklist

- [x] Automatyczna klasyfikacja (min. 5 klas) - **36 klas**
- [x] LAS/LAZ input/output
- [x] IFC output (bonus)
- [x] Raport jakosci (TXT + JSON)
- [x] Web UI + CLI
- [x] Skalowalnosc (tiling)
- [x] Docker
- [x] Instrukcja uruchomienia

---

## Autorzy

HackNation 2025 - Centralny Port Komunikacyjny

## Licencja

MIT License
