"""
Klasyfikatory dla 45 klas infrastruktury CPK

Wszystkie klasyfikatory zebrane w jednym module dla łatwego zarządzania.
Używają dekoratora @register_classifier do automatycznej rejestracji.

KATEGORIE:
- ASPRS Standard (2-18): Ground, Vegetation, Buildings, etc.
- Railway Infrastructure (19-23): Rails, Platforms, Power lines, etc.
- Road Infrastructure (30-38): Roads, Sidewalks, Signs, etc.
- BIM Elements (40-47): Walls, Roofs, Columns, etc.
- Custom CPK (64-67): Telecom, Sewage, Temporary elements, etc.
"""

import numpy as np
from typing import Dict, Optional
from .base import BaseClassifier, register_classifier


# ============================================================================
# ASPRS STANDARD CLASSES (2-18)
# ============================================================================

@register_classifier(class_id=2)
class GroundClassifier(BaseClassifier):
    """Grunt - gołe podłoże bez obiektów"""

    def __init__(self):
        super().__init__(class_id=2, class_name="Grunt", priority=20)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        mask = np.zeros(len(coords), dtype=bool)

        # Strefy A i B (teren)
        zone_mask = np.isin(height_zones, ['A', 'B'])

        # Wysoka płaskość
        planarity_mask = features['planarity'] > 0.8

        # Niska chropowatość
        roughness_mask = features['roughness'] < 0.15

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.9

        # Brak zieleni (jeśli są kolory)
        if colors is not None:
            ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
            vegetation_mask = ndvi < 0.1
        else:
            vegetation_mask = True

        mask = zone_mask & planarity_mask & roughness_mask & horizontal_mask & vegetation_mask
        return mask


@register_classifier(class_id=3)
class LowVegetationClassifier(BaseClassifier):
    """Niska roślinność (< 0.5m)"""

    def __init__(self):
        super().__init__(class_id=3, class_name="Niska roślinność", priority=50)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        if colors is None:
            return np.zeros(len(coords), dtype=bool)

        # Strefa B (poziom terenu)
        zone_mask = height_zones == 'B'

        # NDVI - zielony kolor
        ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
        green_mask = ndvi > 0.1

        # Umiarkowana chropowatość
        roughness_mask = (features['roughness'] > 0.1) & (features['roughness'] < 0.4)

        mask = zone_mask & green_mask & roughness_mask
        return mask


@register_classifier(class_id=4)
class MediumVegetationClassifier(BaseClassifier):
    """Średnia roślinność (0.5-3m)"""

    def __init__(self):
        super().__init__(class_id=4, class_name="Średnia roślinność", priority=50)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        if colors is None:
            return np.zeros(len(coords), dtype=bool)

        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # NDVI
        ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
        green_mask = ndvi > 0.15

        # Wyższa chropowatość
        roughness_mask = features['roughness'] > 0.2

        # Wysoka wariancja wysokości
        variance_mask = features['height_variance'] > 0.5

        mask = zone_mask & green_mask & roughness_mask & variance_mask
        return mask


@register_classifier(class_id=5)
class HighVegetationClassifier(BaseClassifier):
    """Wysoka roślinność - drzewa (> 3m)"""

    def __init__(self):
        super().__init__(class_id=5, class_name="Wysoka roślinność", priority=50)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        if colors is None:
            return np.zeros(len(coords), dtype=bool)

        # Strefy C i D (wysoko)
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # NDVI
        ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
        green_mask = ndvi > 0.15

        # Bardzo wysoka chropowatość
        roughness_mask = features['roughness'] > 0.3

        # Bardzo wysoka wariancja wysokości
        variance_mask = features['height_variance'] > 1.0

        mask = zone_mask & green_mask & roughness_mask & variance_mask
        return mask


@register_classifier(class_id=6)
class BuildingClassifier(BaseClassifier):
    """Budynki - główne struktury"""

    def __init__(self):
        super().__init__(class_id=6, class_name="Budynki", priority=10)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B, C, D (ponad terenem)
        zone_mask = np.isin(height_zones, ['B', 'C', 'D'])

        # Wysoka płaskość
        planarity_mask = features['planarity'] > 0.6

        # Niska chropowatość
        roughness_mask = features['roughness'] < 0.2

        mask = zone_mask & planarity_mask & roughness_mask
        return mask


@register_classifier(class_id=7)
class NoiseClassifier(BaseClassifier):
    """Szum - izolowane punkty, outliers"""

    def __init__(self):
        super().__init__(class_id=7, class_name="Szum", priority=99)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Bardzo niska gęstość (izolowane punkty)
        density_mask = features['density'] < 5

        mask = density_mask
        return mask


@register_classifier(class_id=9)
class WaterClassifier(BaseClassifier):
    """Woda"""

    def __init__(self):
        super().__init__(class_id=9, class_name="Woda", priority=40)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa A (poniżej terenu)
        zone_mask = height_zones == 'A'

        # Bardzo płasko
        planarity_mask = features['planarity'] > 0.85

        # Bardzo gładko
        roughness_mask = features['roughness'] < 0.05

        # Niska intensywność (woda absorbuje laser)
        if intensity is not None:
            intensity_mask = intensity < 0.2
        else:
            intensity_mask = True

        # Ciemny kolor
        if colors is not None:
            brightness = colors.mean(axis=1)
            dark_mask = brightness < 0.3
        else:
            dark_mask = True

        mask = zone_mask & planarity_mask & roughness_mask & intensity_mask & dark_mask
        return mask


@register_classifier(class_id=17)
class BridgeClassifier(BaseClassifier):
    """Mosty - NAJWYŻSZY PRIORYTET"""

    def __init__(self):
        super().__init__(class_id=17, class_name="Most", priority=1)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D (wysoko)
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Bardzo wysoka płaskość
        planarity_mask = features['planarity'] > 0.75

        # Gładka powierzchnia
        roughness_mask = features['roughness'] < 0.12

        mask = zone_mask & planarity_mask & roughness_mask
        return mask


@register_classifier(class_id=18)
class RailClassifier(BaseClassifier):
    """Tory kolejowe"""

    def __init__(self):
        super().__init__(class_id=18, class_name="Tory kolejowe", priority=15)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Wysoka liniowość
        linearity_mask = features['linearity'] > 0.7

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.85

        mask = zone_mask & linearity_mask & horizontal_mask
        return mask


# ============================================================================
# RAILWAY INFRASTRUCTURE (19-23)
# ============================================================================

@register_classifier(class_id=19)
class PowerLinesClassifier(BaseClassifier):
    """Linie wysokiego napięcia"""

    def __init__(self):
        super().__init__(class_id=19, class_name="Linie wysokiego napięcia", priority=30)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa D (bardzo wysoko)
        zone_mask = height_zones == 'D'

        # Ekstremalna liniowość
        linearity_mask = features['linearity'] > 0.95

        mask = zone_mask & linearity_mask
        return mask


@register_classifier(class_id=20)
class TractionPolesClassifier(BaseClassifier):
    """Słupy trakcyjne"""

    def __init__(self):
        super().__init__(class_id=20, class_name="Słupy trakcyjne", priority=35)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Wysoka liniowość (pionowy słup)
        linearity_mask = features['linearity'] > 0.8

        # Wertykalność
        vertical_mask = features['verticality'] > 0.85

        mask = zone_mask & linearity_mask & vertical_mask
        return mask


@register_classifier(class_id=21)
class PlatformClassifier(BaseClassifier):
    """Perony kolejowe"""

    def __init__(self):
        super().__init__(class_id=21, class_name="Peron kolejowy", priority=25)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Płaskość
        planarity_mask = features['planarity'] > 0.8

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.9

        mask = zone_mask & planarity_mask & horizontal_mask
        return mask


@register_classifier(class_id=22)
class RailwaySignsClassifier(BaseClassifier):
    """Znaki kolejowe"""

    def __init__(self):
        super().__init__(class_id=22, class_name="Znak kolejowy", priority=45)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Płaskość (małe płaskie elementy)
        planarity_mask = features['planarity'] > 0.6

        # Wysoka intensywność (odblaskowe)
        if intensity is not None:
            intensity_mask = intensity > 0.6
        else:
            intensity_mask = True

        mask = zone_mask & planarity_mask & intensity_mask
        return mask


@register_classifier(class_id=23)
class RailwayFencesClassifier(BaseClassifier):
    """Ogrodzenia kolejowe"""

    def __init__(self):
        super().__init__(class_id=23, class_name="Ogrodzenie kolejowe", priority=55)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Liniowość
        linearity_mask = features['linearity'] > 0.6

        # Wertykalność
        vertical_mask = features['verticality'] > 0.7

        mask = zone_mask & linearity_mask & vertical_mask
        return mask


# ============================================================================
# ROAD INFRASTRUCTURE (30-38)
# ============================================================================

@register_classifier(class_id=30)
class RoadSurfaceClassifier(BaseClassifier):
    """Nawierzchnia drogi - WYSOKI PRIORYTET"""

    def __init__(self):
        super().__init__(class_id=30, class_name="Nawierzchnia drogi", priority=12)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Bardzo wysoka płaskość
        planarity_mask = features['planarity'] > 0.85

        # Bardzo gładko
        roughness_mask = features['roughness'] < 0.08

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.95

        # Wysoka intensywność (asfalt odbija)
        if intensity is not None:
            intensity_mask = intensity > 0.4
        else:
            intensity_mask = True

        # Szary kolor (niska nasycen)
        if colors is not None:
            saturation = colors.std(axis=1)
            gray_mask = saturation < 0.12
        else:
            gray_mask = True

        mask = zone_mask & planarity_mask & roughness_mask & horizontal_mask & intensity_mask & gray_mask
        return mask


@register_classifier(class_id=31)
class SidewalkClassifier(BaseClassifier):
    """Chodnik"""

    def __init__(self):
        super().__init__(class_id=31, class_name="Chodnik", priority=28)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa B
        zone_mask = height_zones == 'B'

        # Płaskość
        planarity_mask = features['planarity'] > 0.8

        # Gładkość
        roughness_mask = features['roughness'] < 0.12

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.9

        mask = zone_mask & planarity_mask & roughness_mask & horizontal_mask
        return mask


@register_classifier(class_id=32)
class CurbClassifier(BaseClassifier):
    """Krawężnik"""

    def __init__(self):
        super().__init__(class_id=32, class_name="Krawężnik", priority=22)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa B
        zone_mask = height_zones == 'B'

        # Liniowość (krawędź)
        linearity_mask = features['linearity'] > 0.75

        mask = zone_mask & linearity_mask
        return mask


@register_classifier(class_id=33)
class RoadGreenStripClassifier(BaseClassifier):
    """Pas zieleni przydrożnej"""

    def __init__(self):
        super().__init__(class_id=33, class_name="Pas zieleni", priority=52)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        if colors is None:
            return np.zeros(len(coords), dtype=bool)

        # Strefa B
        zone_mask = height_zones == 'B'

        # Chropowatość
        roughness_mask = features['roughness'] > 0.2

        # Zieleń
        ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
        green_mask = ndvi > 0.15

        mask = zone_mask & roughness_mask & green_mask
        return mask


@register_classifier(class_id=34)
class StreetLightingClassifier(BaseClassifier):
    """Oświetlenie uliczne"""

    def __init__(self):
        super().__init__(class_id=34, class_name="Latarnia uliczna", priority=38)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Liniowość pionowa
        linearity_mask = features['linearity'] > 0.8

        # Wertykalność
        vertical_mask = features['verticality'] > 0.85

        mask = zone_mask & linearity_mask & vertical_mask
        return mask


@register_classifier(class_id=35)
class RoadSignsClassifier(BaseClassifier):
    """Znaki drogowe"""

    def __init__(self):
        super().__init__(class_id=35, class_name="Znak drogowy", priority=42)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Płaskość
        planarity_mask = features['planarity'] > 0.65

        # Odblaskowe
        if intensity is not None:
            intensity_mask = intensity > 0.7
        else:
            intensity_mask = True

        # Kolorowe (wysoka nasycenie)
        if colors is not None:
            saturation = colors.std(axis=1)
            colorful_mask = saturation > 0.3
        else:
            colorful_mask = True

        mask = zone_mask & planarity_mask & intensity_mask & colorful_mask
        return mask


@register_classifier(class_id=36)
class RoadBarriersClassifier(BaseClassifier):
    """Bariery drogowe"""

    def __init__(self):
        super().__init__(class_id=36, class_name="Bariera drogowa", priority=48)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy B i C
        zone_mask = np.isin(height_zones, ['B', 'C'])

        # Liniowość
        linearity_mask = features['linearity'] > 0.65

        mask = zone_mask & linearity_mask
        return mask


@register_classifier(class_id=37)
class BusStopsClassifier(BaseClassifier):
    """Przystanki autobusowe"""

    def __init__(self):
        super().__init__(class_id=37, class_name="Przystanek autobusowy", priority=58)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa B
        zone_mask = height_zones == 'B'

        # Płaskość
        planarity_mask = features['planarity'] > 0.75

        mask = zone_mask & planarity_mask
        return mask


@register_classifier(class_id=38)
class TrafficLightsClassifier(BaseClassifier):
    """Sygnalizacja świetlna"""

    def __init__(self):
        super().__init__(class_id=38, class_name="Sygnalizacja świetlna", priority=44)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Liniowość pionowa
        linearity_mask = features['linearity'] > 0.75

        # Wertykalność
        vertical_mask = features['verticality'] > 0.8

        mask = zone_mask & linearity_mask & vertical_mask
        return mask


# ============================================================================
# BIM / BUILDING ELEMENTS (40-47)
# ============================================================================

@register_classifier(class_id=40)
class ExternalWallsClassifier(BaseClassifier):
    """Ściany zewnętrzne"""

    def __init__(self):
        super().__init__(class_id=40, class_name="Ściana zewnętrzna", priority=62)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Płaskość
        planarity_mask = features['planarity'] > 0.7

        # Wertykalność
        vertical_mask = features['verticality'] > 0.8

        mask = planarity_mask & vertical_mask
        return mask


@register_classifier(class_id=41)
class RoofsClassifier(BaseClassifier):
    """Dachy"""

    def __init__(self):
        super().__init__(class_id=41, class_name="Dach", priority=18)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Płaskość
        planarity_mask = features['planarity'] > 0.75

        mask = zone_mask & planarity_mask
        return mask


@register_classifier(class_id=42)
class FoundationsClassifier(BaseClassifier):
    """Fundamenty"""

    def __init__(self):
        super().__init__(class_id=42, class_name="Fundament", priority=65)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy A i B
        zone_mask = np.isin(height_zones, ['A', 'B'])

        # Płaskość
        planarity_mask = features['planarity'] > 0.7

        mask = zone_mask & planarity_mask
        return mask


@register_classifier(class_id=43)
class ColumnsClassifier(BaseClassifier):
    """Kolumny"""

    def __init__(self):
        super().__init__(class_id=43, class_name="Kolumna", priority=68)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Bardzo wysoka liniowość
        linearity_mask = features['linearity'] > 0.85

        # Wertykalność
        vertical_mask = features['verticality'] > 0.9

        # Kulistość (cylindryczna)
        sphericity_mask = features['sphericity'] > 0.5

        mask = linearity_mask & vertical_mask & sphericity_mask
        return mask


@register_classifier(class_id=44)
class BeamsClassifier(BaseClassifier):
    """Belki"""

    def __init__(self):
        super().__init__(class_id=44, class_name="Belka", priority=70)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Liniowość
        linearity_mask = features['linearity'] > 0.8

        # Horyzontalność
        horizontal_mask = features['horizontality'] > 0.8

        mask = linearity_mask & horizontal_mask
        return mask


@register_classifier(class_id=45)
class PipesInstallationsClassifier(BaseClassifier):
    """Rury / instalacje"""

    def __init__(self):
        super().__init__(class_id=45, class_name="Rura / instalacja", priority=72)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Kulistość (cylindryczna)
        sphericity_mask = features['sphericity'] > 0.6

        # Liniowość
        linearity_mask = features['linearity'] > 0.7

        mask = sphericity_mask & linearity_mask
        return mask


@register_classifier(class_id=46)
class WindowsDoorsClassifier(BaseClassifier):
    """Okna / drzwi"""

    def __init__(self):
        super().__init__(class_id=46, class_name="Okno / drzwi", priority=75)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Złożona detekcja - na razie pomijamy
        return np.zeros(len(coords), dtype=bool)


@register_classifier(class_id=47)
class RailingsClassifier(BaseClassifier):
    """Balustrady"""

    def __init__(self):
        super().__init__(class_id=47, class_name="Balustrada", priority=78)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Liniowość
        linearity_mask = features['linearity'] > 0.6

        mask = linearity_mask
        return mask


# ============================================================================
# CUSTOM CPK CLASSES (64-67)
# ============================================================================

@register_classifier(class_id=64)
class TelecomPolesClassifier(BaseClassifier):
    """Słupy telekomunikacyjne"""

    def __init__(self):
        super().__init__(class_id=64, class_name="Słup telekomunikacyjny", priority=82)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefy C i D
        zone_mask = np.isin(height_zones, ['C', 'D'])

        # Liniowość
        linearity_mask = features['linearity'] > 0.8

        # Wertykalność
        vertical_mask = features['verticality'] > 0.85

        mask = zone_mask & linearity_mask & vertical_mask
        return mask


@register_classifier(class_id=65)
class SewagePipesClassifier(BaseClassifier):
    """Rurociągi kanalizacyjne"""

    def __init__(self):
        super().__init__(class_id=65, class_name="Rurociąg kanalizacyjny", priority=85)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Strefa A (podziemie)
        zone_mask = height_zones == 'A'

        # Kulistość (cylindryczna)
        sphericity_mask = features['sphericity'] > 0.7

        # Liniowość
        linearity_mask = features['linearity'] > 0.75

        mask = zone_mask & sphericity_mask & linearity_mask
        return mask


@register_classifier(class_id=66)
class TemporaryElementsClassifier(BaseClassifier):
    """Elementy tymczasowe (rusztowania, itp.)"""

    def __init__(self):
        super().__init__(class_id=66, class_name="Element tymczasowy", priority=88)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Złożona geometria (niska płaskość)
        complex_mask = features['planarity'] < 0.5

        # Wysoka wariancja wysokości
        variance_mask = features['height_variance'] > 0.5

        mask = complex_mask & variance_mask
        return mask


@register_classifier(class_id=67)
class ConstructionDetailsClassifier(BaseClassifier):
    """Detale budowlane - specjalne elementy"""

    def __init__(self):
        super().__init__(class_id=67, class_name="Detal budowlany", priority=92)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Catch-all dla unikalnych elementów - na razie pomijamy
        return np.zeros(len(coords), dtype=bool)


# ============================================================================
# UNCLASSIFIED (1) - Domyślna klasa
# ============================================================================

@register_classifier(class_id=1)
class UnclassifiedClassifier(BaseClassifier):
    """Niesklasyfikowane - domyślna klasa dla nierozpoznanych punktów"""

    def __init__(self):
        super().__init__(class_id=1, class_name="Niesklasyfikowane", priority=100)

    def classify(self, coords, features, height_zones, colors=None, intensity=None):
        # Zawsze zwraca False - punkty są klasyfikowane jako 1 jeśli nic innego nie pasuje
        return np.zeros(len(coords), dtype=bool)
