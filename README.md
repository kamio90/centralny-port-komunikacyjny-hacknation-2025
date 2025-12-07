# Chmura+ v2.0

**HackNation 2025 - Centralny Port Komunikacyjny**

Automatyczna klasyfikacja elementow infrastruktury na podstawie chmur punktow LAS/LAZ z zaawansowanymi modulami ML, Railway i BIM.

![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

---

## Funkcjonalnosci

### Core
- **36 klas infrastruktury**: ASPRS, Railway, Road, BIM, CPK Custom
- **Wysoka wydajnosc**: ~400k punktow/s w trybie szybkim
- **Duze pliki**: Obsluga 10GB+ (tiling, memory-mapped I/O)
- **Raporty**: TXT + JSON + IFC
- **Interfejs**: Web UI (Streamlit) + CLI

### Modul ML (Machine Learning)
- **Random Forest** - klasyfikator lesny z automatycznym strojeniem
- **PointNet** - siec neuronowa dla chmur punktow (PyTorch)
- **Ensemble** - laczenie wielu modeli (voting, stacking, boosting)
- **Active Learning** - uczenie aktywne z selekcja probek
- **Auto-tuning** - automatyczne dostrajanie hiperparametrow
- **Post-processing** - wygladzanie, usuwanie szumu, laczenie klas
- **Model Comparison** - porownywanie i benchmarking modeli

### Modul Railway (Kolej)
- **Catenary Detection** - detekcja sieci trakcyjnej
- **Track Extraction** - ekstrakcja torow kolejowych
- **Pole Detection** - wykrywanie slupow trakcyjnych
- **Signal Detection** - detekcja sygnalow kolejowych
- **Infrastructure Report** - raporty infrastruktury kolejowej

### Modul BIM (Building Information Modeling)
- **Building Extraction** - ekstrakcja budynkow z DBSCAN i RANSAC
- **Geometry Analyzer** - analiza 3D (AABB, OBB, Convex Hull, PCA)
- **LOD Classification** - klasyfikacja poziomu szczegolowosci (LOD 100-500)
- **Clash Detection** - detekcja kolizji przestrzennych
- **IFC Export** - eksport do formatu BIM (IFC-SPF, JSON, XML)

### Analiza
- **Terrain Analysis** - analiza terenu (spadki, ekspozycja)
- **Volume Calculator** - obliczenia objetosci
- **Railway Clearance** - skrajnia kolejowa

---

## Szybki start

### Wymagania

- Python 3.9+ (zalecany 3.11)
- RAM: 8GB minimum (16GB+ zalecane dla duzych plikow)
- GPU: opcjonalny (PyTorch z CUDA/MPS dla PointNet)

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

# Z eksportem IFC
python cli.py data/input.las output/classified.las --ifc output/model.ifc
```

---

## Docker

### Szybki start z Docker Compose

```bash
# Uruchom
docker-compose up -d

# Zatrzymaj
docker-compose down

# Rebuild (po zmianach)
docker-compose build --no-cache && docker-compose up -d
```

Aplikacja: http://localhost:8501

### Reczny build i uruchomienie

```bash
# Build
docker build -t cpk-classifier .

# Uruchom Web UI
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --name cpk \
  cpk-classifier

# Uruchom CLI
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  cpk-classifier \
  python cli.py /app/data/input.las /app/output/output.las --report /app/output/raport.json

# Zatrzymaj
docker stop cpk && docker rm cpk
```

### Logi

```bash
docker logs -f cpk
# lub
docker-compose logs -f
```

### Testowanie z wlasnym plikiem LAS/LAZ

**Sposob 1: Upload przez interfejs webowy (najlatwiejszy)**

1. Uruchom aplikacje: `docker-compose up -d`
2. Otworz http://localhost:8501
3. Przejdz do zakladki **"Wczytaj plik"**
4. Przeciagnij swoj plik LAS/LAZ lub kliknij "Browse files"

**Sposob 2: Przez folder data/ (dla duzych plikow)**

1. Skopiuj swoj plik do folderu `data/`:
```bash
cp /sciezka/do/twojego/pliku.las ./data/
```

2. Uruchom aplikacje:
```bash
docker-compose up -d
```

3. W aplikacji wybierz plik z listy dostepnych

**Struktura folderow:**
```
cpk-clasificator/
├── data/          ← tutaj wrzuc swoje pliki LAS/LAZ
├── output/        ← tutaj pojawia sie wyniki
├── models/        ← zapisane modele ML
└── ...
```

**Tryb DEMO (bez wlasnego pliku)**

Jesli nie masz pliku LAS/LAZ, uzyj **trybu demo** w zakladce "Wczytaj plik" - aplikacja wygeneruje syntetyczna chmure punktow do testowania.

---

## Interfejs Web (8 zakladek)

| Zakladka | Opis |
|----------|------|
| **Wczytaj plik** | Upload plikow LAS/LAZ, tryb demo |
| **Podglad** | Wizualizacja 3D chmury punktow |
| **Hackathon** | Szybka klasyfikacja dla hackathonu |
| **Analiza** | Analiza terenu, objetosci, statystyki |
| **ML** | Klasyfikatory ML (RF, PointNet, Ensemble) |
| **Railway** | Analiza infrastruktury kolejowej |
| **BIM** | Analiza BIM, eksport IFC, kolizje |
| **Klasyfikacja** | Pelna klasyfikacja 36 klas |

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
| 19 | Linie energetyczne (siec trakcyjna) |
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
                                           ↓
                                    ML Pipeline (opcjonalnie)
                                           ↓
                                    Railway/BIM Analysis
```

### Moduly

```
cpk-clasificator/
├── app.py                    # Web UI (Streamlit)
├── cli.py                    # Command Line Interface
├── src/
│   ├── config.py             # Konfiguracja
│   ├── ui/                   # Komponenty UI
│   │   ├── styles.py
│   │   └── components/
│   │       ├── file_loader.py
│   │       ├── preview.py
│   │       ├── classification.py
│   │       ├── hackathon_classification.py
│   │       ├── analysis.py
│   │       ├── ml_classifier.py
│   │       ├── railway_analyzer.py
│   │       └── bim_analyzer.py
│   └── v2/
│       ├── core/             # LASLoader, LASWriter, TilingEngine
│       ├── features/         # GeometricFeatureExtractor
│       ├── classifiers/      # 36 klasyfikatorow
│       ├── algorithms/       # CSF, HAG, SOR, RANSAC
│       ├── pipeline/         # ClassificationPipeline
│       ├── analysis/         # Terrain, Volume, Clearance
│       ├── exporters/        # GeoJSON, HTML Viewer
│       ├── ml/               # Random Forest, PointNet, Ensemble
│       ├── railway/          # Catenary, Track, Pole, Signal
│       └── bim/              # Building, Clash, IFC, LOD, Geometry
├── data/                     # Pliki wejsciowe
├── output/                   # Wyniki
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-docker.txt
```

### Algorytmy

| Algorytm | Zastosowanie |
|----------|--------------|
| **CSF** (Cloth Simulation Filter) | Detekcja gruntu |
| **HAG** (Height Above Ground) | Strefy wysokosci |
| **PCA** | Cechy geometryczne (planarity, linearity, sphericity) |
| **NDVI** | Wykrywanie roslinnosci z RGB |
| **RANSAC** | Detekcja plaszczyzn, dachow |
| **DBSCAN** | Klasteryzacja budynkow |
| **SOR** (Statistical Outlier Removal) | Usuwanie szumu |
| **Random Forest** | Klasyfikacja ML |
| **PointNet** | Klasyfikacja deep learning |
| **Convex Hull** | Analiza geometrii 3D |

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
  },
  "ml_metrics": {
    "model": "ensemble",
    "accuracy": 0.94,
    "f1_score": 0.92
  },
  "bim": {
    "buildings_detected": 15,
    "lod_level": "LOD300",
    "clashes_found": 3
  }
}
```

### IFC (Industry Foundation Classes)
Elementy BIM wyekstrahowane z klasyfikacji:
- Budynki (IfcBuilding)
- Teren (IfcSite)
- Infrastruktura (IfcBuildingElementProxy)

---

## API Modulow

### ML
```python
from src.v2.ml import RandomForestPointClassifier, PointNetTrainer, EnsembleClassifier

# Random Forest
rf = RandomForestPointClassifier()
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# PointNet (wymaga PyTorch)
from src.v2.ml import PointNetConfig, is_torch_available
if is_torch_available():
    config = PointNetConfig(num_classes=10, num_points=1024)
    trainer = PointNetTrainer(config)

# Ensemble
ensemble = EnsembleClassifier(method='voting')
ensemble.add_model(rf)
predictions = ensemble.predict(X_test)
```

### Railway
```python
from src.v2.railway import CatenaryDetector, TrackExtractor, InfrastructureReporter

# Detekcja sieci trakcyjnej
catenary = CatenaryDetector(coords, classifications)
wires = catenary.detect()

# Raport
reporter = InfrastructureReporter(coords, classifications)
html = reporter.generate_html()
```

### BIM
```python
from src.v2.bim import BuildingExtractor, ClashDetector, IFCExporter, LODClassifier

# Ekstrakcja budynkow
extractor = BuildingExtractor(coords)
buildings = extractor.extract()

# Detekcja kolizji
detector = ClashDetector()
detector.add_buildings(buildings)
clashes = detector.detect()

# Eksport IFC
exporter = IFCExporter(project_name="CPK")
exporter.add_buildings(buildings)
exporter.export("model.ifc")

# Klasyfikacja LOD
lod = LODClassifier(coords)
result = lod.classify()
print(f"LOD Level: {result.level}")
```

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
- [x] IFC output (bonus) - **pelny eksport BIM**
- [x] Raport jakosci (TXT + JSON)
- [x] Web UI + CLI
- [x] Skalowalnosc (tiling)
- [x] Docker + Docker Compose
- [x] Instrukcja uruchomienia
- [x] **Modul ML** (Random Forest, PointNet, Ensemble)
- [x] **Modul Railway** (Catenary, Track, Pole, Signal)
- [x] **Modul BIM** (Building, Clash, LOD, Geometry)

---

## Troubleshooting

### Docker - problemy z pamiecia
```bash
# Zwieksz limit pamieci w Docker Desktop
# Settings → Resources → Memory: 8GB+
```

### Open3D na ARM (Mac M1/M2/M3)
```bash
# Jesli problemy z instalacja:
pip install open3d --no-cache-dir
```

### Streamlit - blad portu
```bash
# Zmien port
streamlit run app.py --server.port 8502
```

---

## Autorzy

HackNation 2025 - Centralny Port Komunikacyjny - Zespol Chmura+

## Licencja

MIT License
