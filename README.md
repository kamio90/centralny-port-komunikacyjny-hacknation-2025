# Chmura+ v2.0

**HackNation 2025 - Centralny Port Komunikacyjny**

Automatyczna klasyfikacja elementów infrastruktury na podstawie chmur punktów LAS/LAZ z wykorzystaniem geometrycznych cech PCA i metodologii BIM.

![Chmura+ Logo](assets/logo_chmura.png)

![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🎯 Główne funkcjonalności

### Cechy systemu
- **🚀 Wysoka wydajność**: Przetwarzanie 277M punktów w ~10-12 minut (tryb DEMO)
- **🎨 45 klas infrastruktury**: ASPRS, Railway, Road, BIM, CPK Custom
- **🧩 Modularna architektura**: Łatwe dodawanie nowych klasyfikatorów
- **⚡ Przetwarzanie równoległe**: Thread-safe z ThreadPoolExecutor (4 wątki)
- **📊 Raporty jakości**: Szczegółowe statystyki i wykresy (TXT + JSON)
- **💾 Obsługa dużych plików**: 10GB+ bez problemu (pamięć mapowana)
- **🌍 Interface w języku polskim**: Pełne wsparcie Unicode

### Techniczne highlights
- **Spatial Tiling**: Automatyczne dzielenie na kafelki 5-15m (adaptacyjne do gęstości)
- **PCA Feature Extraction**: Planarity, linearity, sphericity, NDVI, brightness
- **Vectorized Processing**: NumPy dla maksymalnej wydajności
- **Real-time Progress**: Stabilne ETA bazujące na punktach/sekunda
- **LAS Format Compatibility**: Auto-remapping klas >31 dla zgodności z LAS 1.2/1.3

---

## 📋 Klasyfikacja - 45 klas

### ASPRS Standard (2-18)
| ID | Nazwa | Opis |
|----|-------|------|
| 1 | Nieklasyfikowane | Punkty niesklasyfikowane |
| 2 | Grunt | Powierzchnia terenu |
| 3 | Niska roślinność | Trawa, krzewy <0.5m |
| 4 | Średnia roślinność | Krzewy 0.5-2m |
| 5 | Wysoka roślinność | Drzewa >2m |
| 6 | Budynki | Struktury budowlane |
| 7 | Szum niski | Artefakty blisko terenu |
| 9 | Woda | Powierzchnie wodne |
| 13 | Mosty | Konstrukcje mostowe |
| 17 | Naziemne obiekty | Konstrukcje techniczne |
| 18 | Wysokie szumy | Artefakty wysokościowe |

### Railway (19-23)
| ID | Nazwa | Opis |
|----|-------|------|
| 19 | Szyny kolejowe | Tory kolejowe |
| 20 | Podkłady kolejowe | Podkłady i podsypka |
| 21 | Trakcja kolejowa | Sieć trakcyjna |
| 22 | Perony | Perony stacji |
| 23 | Infrastruktura kolejowa | Inne elementy |

### Road (30-38)
| ID | Nazwa | Opis |
|----|-------|------|
| 30 | Droga - jezdnia | Nawierzchnia drogowa |
| 31 | Droga - chodnik | Chodniki |
| 32 | Droga - krawężnik | Krawężniki |
| 33 | Droga - oznakowanie | Linie na jezdni |
| 34 | Droga - bariery | Bariery ochronne |
| 35 | Droga - znaki | Znaki drogowe |
| 36 | Droga - słupy | Słupy oświetleniowe |
| 37 | Droga - sygnalizacja | Światła |
| 38 | Droga - inne | Inne elementy |

### BIM Infrastructure (40-47)
| ID | Nazwa | Opis |
|----|-------|------|
| 40 | BIM - fundamenty | Konstrukcje fundamentowe |
| 41 | BIM - ściany | Ściany budowli |
| 42 | BIM - dachy | Konstrukcje dachowe |
| 43 | BIM - instalacje | MEP (HVAC, elektryka) |
| 44 | BIM - konstrukcje stalowe | Elementy stalowe |
| 45 | BIM - elementy prefabrykowane | Prefabrykaty |
| 46 | BIM - wykończenie | Elewacje, okładziny |
| 47 | BIM - tereny zielone | Landscaping BIM |

### CPK Custom (64-67)
| ID | Nazwa | Opis |
|----|-------|------|
| 64 | CPK - Terminal | Terminale lotniskowe |
| 65 | CPK - Runway | Pasy startowe |
| 66 | CPK - Rail Hub | Węzeł kolejowy |
| 67 | CPK - Parking | Parkingi wielopoziomowe |

---

## 🚀 Szybki start

### Wymagania systemowe
- **Python**: 3.9 lub nowszy
- **RAM**: Minimum 8GB (zalecane 16GB+ dla dużych chmur)
- **Dysk**: Miejsce na pliki LAS/LAZ (10GB+ dla pełnych zbiorów)
- **System**: macOS, Linux, Windows

### Instalacja

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/your-repo/cpk-clasificator.git
cd cpk-clasificator

# 2. Utwórz wirtualne środowisko
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# LUB
venv\Scripts\activate  # Windows

# 3. Zainstaluj zależności
pip install -r requirements.txt
```

### Uruchomienie

```bash
# Uruchom aplikację Streamlit
streamlit run app.py
```

Aplikacja uruchomi się w przeglądarce pod adresem: http://localhost:8501

### Użycie - krok po kroku

1. **Wczytaj plik**:
   - Zakładka "Wczytaj plik"
   - Wybierz plik z listy (folder `data/`) LUB podaj własną ścieżkę
   - System wyświetli informacje o pliku (liczba punktów, rozmiar, kolory RGB)

2. **Uruchom klasyfikację**:
   - Zakładka "Klasyfikacja"
   - Ustaw opcje:
     - Tryb DEMO (szybszy) vs Normalny (dokładniejszy)
     - Liczba wątków (1-8)
   - Kliknij "ROZPOCZNIJ KLASYFIKACJĘ"

3. **Pobierz wyniki**:
   - Sklasyfikowany plik LAS
   - Raport TXT (szczegółowe statystyki)
   - Raport JSON (programatyczne przetwarzanie)

---

## 📊 Wydajność

### Testowano na: Apple M4 Max, 64GB RAM

| Metryka | Tryb Normal | Tryb DEMO |
|---------|-------------|-----------|
| **Liczba punktów** | 277,529,209 | 277,529,209 |
| **Czas przetwarzania** | ~20-25 min | ~10-12 min |
| **Prędkość** | ~180k pkt/s | ~400k pkt/s |
| **Liczba kafelków** | ~100-150 | ~28 |
| **Sample rate PCA** | 0.5% | 0.02% |
| **Pamięć (peak)** | ~8-12GB | ~6-8GB |

### Optymalizacje
- **Vectorized tiling**: Pojedyncze przejście przez wszystkie punkty (100x szybsze)
- **PCA sampling**: 0.02% w trybie DEMO (200x szybsze feature extraction)
- **Parallel processing**: ThreadPoolExecutor z 4 wątkami
- **Memory-mapped I/O**: laspy z optymalizacją pamięci

---

## 🏗️ Architektura

### Struktura katalogów

```
cpk-clasificator/
├── app.py                          # Streamlit UI (główna aplikacja)
├── src/
│   └── v2/                         # Nowa architektura v2.0
│       ├── __init__.py
│       ├── core/                   # Podstawowe operacje I/O
│       │   ├── las_loader.py       # Wczytywanie LAS/LAZ
│       │   ├── las_writer.py       # Zapis + raporty
│       │   └── tiling_engine.py    # Spatial tiling
│       ├── features/               # Ekstrakcja cech
│       │   └── geometric_features.py  # PCA + kolory
│       ├── classifiers/            # Klasyfikatory
│       │   ├── base.py             # BaseClassifier + Registry
│       │   └── infrastructure_classifiers.py  # 45 klas
│       └── pipeline/               # Orkiestracja
│           └── classification_pipeline.py  # Main pipeline
├── data/                           # Pliki wejściowe LAS/LAZ
├── output/                         # Wyniki klasyfikacji
├── requirements.txt
├── Dockerfile
└── README.md
```

### Design Patterns

**1. Registry Pattern** - Dekorator `@register_classifier`
```python
@register_classifier(class_id=2)
class GroundClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(class_id=2, class_name="Grunt", priority=20)

    def classify(self, coords, features, height_zones, colors, intensity):
        # Logika klasyfikacji
        return mask
```

**2. Pipeline Pattern** - Modularny flow
```
LASLoader → TilingEngine → GeometricFeatureExtractor
    → ClassifierRegistry → LASWriter
```

**3. Thread-Safe Processing** - Każdy kafelek niezależny
- Local KD-trees per tile
- Progress tracking z threading.Lock
- No shared mutable state

---

## 🐳 Docker

### Build image

```bash
docker build -t cpk-classifier .
```

### Uruchom kontener

```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  cpk-classifier
```

Aplikacja będzie dostępna pod: http://localhost:8501

---

## 📖 API (Programatyczne użycie)

### Przykład Python

```python
from src.v2 import ClassificationPipeline

# Utwórz pipeline
pipeline = ClassificationPipeline(
    input_path="data/moja_chmura.las",
    output_path="output/wynik.las",
    n_threads=4,
    demo_mode=True  # Szybki tryb
)

# Uruchom z callbackiem
def progress(info):
    print(f"Postęp: {info['progress_pct']:.1f}% | ETA: {info['eta_seconds']:.0f}s")

stats = pipeline.run(progress_callback=progress)

print(f"Przetworzono {stats['n_points']:,} punktów w {stats['processing_time']:.1f}s")
print(f"Prędkość: {stats['points_per_second']:,.0f} pkt/s")
```

---

## 🧪 Dodawanie nowych klasyfikatorów

### Szablon klasyfikatora

```python
from src.v2.classifiers import register_classifier, BaseClassifier

@register_classifier(class_id=100)
class MyCustomClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(
            class_id=100,
            class_name="Moja klasa",
            priority=50  # Wyższa liczba = wyższy priorytet
        )

    def classify(self, coords, features, height_zones, colors, intensity):
        """
        Args:
            coords: (N, 3) XYZ
            features: dict {'planarity': ..., 'linearity': ..., ...}
            height_zones: (N,) strefy A/B/C/D (0/1/2/3)
            colors: (N, 3) RGB [0-1] lub None
            intensity: (N,) [0-1] lub None

        Returns:
            (N,) boolean mask - True = należy do tej klasy
        """
        # Twoja logika klasyfikacji
        mask = (features['planarity'] > 0.8) & (height_zones == 0)

        return mask
```

System automatycznie wykryje i zarejestruje nowy klasyfikator!

---

## 📝 Format raportu jakości

### TXT Format

```
============================================================
RAPORT KLASYFIKACJI CHMURY PUNKTÓW
============================================================
Całkowita liczba punktów: 277,529,209
Liczba wykrytych klas: 15

Rozkład klasyfikacji:
------------------------------------------------------------
  [ 2] Grunt                         125,234,567 (45.12%)
  [ 6] Budynki                        67,891,234 (24.46%)
  [30] Droga - jezdnia                34,567,890 (12.45%)
  ...
============================================================
```

### JSON Format

```json
{
  "metadata": {
    "plik_wejsciowy": "chmura.las",
    "plik_wyjsciowy": "chmura_classified.las",
    "czas_przetwarzania_s": 645.3,
    "predkosc_pkt_s": 430234,
    "liczba_kafelkow": 28,
    "tryb_demo": true,
    "liczba_watkow": 4
  },
  "statystyki": {
    "calkowita_liczba_punktow": 277529209,
    "sklasyfikowane": 265432198,
    "nieklasyfikowane": 12097011,
    "wykryte_klasy": 15
  },
  "rozklad_klas": [
    {
      "id": 2,
      "nazwa": "Grunt",
      "liczba": 125234567,
      "procent": 45.12
    },
    ...
  ]
}
```

---

## 🔧 Troubleshooting

### Problem: Błąd pamięci (MemoryError)

**Rozwiązanie**:
- Włącz tryb DEMO (mniejsze kafelki, mniej próbek PCA)
- Zmniejsz liczbę wątków (n_threads=2)
- Użyj maszyny z większą pamięcią RAM

### Problem: Długi czas przetwarzania

**Rozwiązanie**:
- Włącz tryb DEMO (0.02% sampling vs 0.5%)
- Zwiększ liczbę wątków (jeśli masz więcej rdzeni)
- Podziel plik na mniejsze fragmenty

### Problem: Klasy > 31 w pliku LAS

**Rozwiązanie**: System automatycznie remapuje klasy 32-67 do zakresu 19-31 (User Defined) dla zgodności z LAS 1.2/1.3

---

## 📚 Wymagania hakatonu - Checklist

- ✅ **Automatyczna klasyfikacja**: 45 klas infrastruktury
- ✅ **Obsługa LAS/LAZ**: Wczytywanie i zapis z zachowaniem metadanych
- ✅ **Tiling dla dużych plików**: Adaptacyjne kafelkowanie 5-300m
- ✅ **Raport jakości**: TXT + JSON z pełnymi statystykami
- ✅ **Web interface**: Streamlit w języku polskim
- ✅ **README z instrukcjami**: Ten dokument
- ✅ **Dockerfile**: Gotowy do deploymentu
- ✅ **Modularność**: Łatwe dodawanie nowych klas
- ✅ **Wydajność**: <15 minut dla 277M punktów
- ✅ **BIM Methodology**: Klasy 40-47 zgodne z BIM

---

## 👥 Autorzy

**HackNation 2025 - Centralny Port Komunikacyjny**

## 📄 Licencja

MIT License - Szczegóły w pliku `LICENSE`

---

**Zbudowano dla HackNation 2025** 🏗️🚀
