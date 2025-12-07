"""
IFC Export - Eksport do formatu Industry Foundation Classes

Format IFC jest standardem BIM do wymiany danych miedzy aplikacjami.

Obsługiwane elementy:
- IfcBuilding - budynki
- IfcSite - teren
- IfcBuildingElementProxy - elementy ogolne
- IfcPointCloud - chmury punktow (IFC 4.3)

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BIMElement:
    """Element BIM do eksportu"""
    guid: str
    name: str
    element_type: str  # building, site, terrain, infrastructure, etc.
    geometry: Dict[str, Any]  # vertices, faces, bounds
    properties: Dict[str, Any]  # atrybuty
    classification: Optional[str] = None  # np. Uniclass code
    location: Optional[np.ndarray] = None
    parent_guid: Optional[str] = None


class IFCExporter:
    """
    Eksporter do formatu IFC

    Generuje pliki IFC z:
    - Budynkow (ekstrakcja z chmury punktow)
    - Terenu (DTM)
    - Infrastruktury
    - Metadanych projektu

    Note: Pełna zgodność z IFC 4.3 wymaga biblioteki ifcopenshell.
    Ten moduł generuje podstawowy format IFC-SPF (STEP Physical File).

    Usage:
        exporter = IFCExporter(project_name="CPK Project")
        exporter.add_building(building)
        exporter.add_terrain(dtm_points)
        exporter.export("output.ifc")
    """

    IFC_VERSION = "IFC4X3"

    def __init__(
        self,
        project_name: str = "CPK Project",
        author: str = "CPK Chmura+",
        organization: str = "HackNation 2025"
    ):
        self.project_name = project_name
        self.author = author
        self.organization = organization

        self.elements: List[BIMElement] = []
        self._guid_counter = 1000

        # IFC entities
        self._entities = []
        self._entity_id = 1

    def _generate_guid(self) -> str:
        """Generuj unikalny GUID"""
        self._guid_counter += 1
        # Simplified GUID for IFC (22 chars base64-like)
        import hashlib
        hash_input = f"{self.project_name}_{self._guid_counter}_{datetime.now().isoformat()}"
        hash_hex = hashlib.md5(hash_input.encode()).hexdigest()[:22]
        return hash_hex.upper()

    def add_building(
        self,
        name: str,
        footprint_vertices: np.ndarray,
        height: float,
        ground_elevation: float,
        properties: Optional[Dict] = None
    ) -> str:
        """
        Dodaj budynek

        Args:
            name: nazwa budynku
            footprint_vertices: (N, 2) wierzcholki obrysu
            height: wysokosc budynku [m]
            ground_elevation: wysokosc podstawy [m]
            properties: dodatkowe wlasciwosci

        Returns:
            GUID elementu
        """
        guid = self._generate_guid()

        # Oblicz centroid
        centroid = footprint_vertices.mean(axis=0)

        element = BIMElement(
            guid=guid,
            name=name,
            element_type="IfcBuilding",
            geometry={
                "type": "extrusion",
                "footprint": footprint_vertices.tolist(),
                "height": height,
                "base_elevation": ground_elevation
            },
            properties=properties or {},
            location=np.array([centroid[0], centroid[1], ground_elevation])
        )

        self.elements.append(element)
        logger.info(f"Added building: {name} (GUID: {guid})")
        return guid

    def add_terrain(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        heights: np.ndarray,
        name: str = "Terrain"
    ) -> str:
        """
        Dodaj model terenu

        Args:
            grid_x: wspolrzedne X siatki
            grid_y: wspolrzedne Y siatki
            heights: macierz wysokosci
            name: nazwa

        Returns:
            GUID elementu
        """
        guid = self._generate_guid()

        # Uproszczenie siatki dla eksportu
        step = max(1, len(grid_x) // 100)

        sampled_x = grid_x[::step]
        sampled_y = grid_y[::step]
        sampled_heights = heights[::step, ::step]

        element = BIMElement(
            guid=guid,
            name=name,
            element_type="IfcSite",
            geometry={
                "type": "terrain_grid",
                "grid_x": sampled_x.tolist(),
                "grid_y": sampled_y.tolist(),
                "heights": sampled_heights.tolist(),
                "resolution": float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0
            },
            properties={
                "original_resolution": float(grid_x[1] - grid_x[0]) if len(grid_x) > 1 else 1.0,
                "grid_size_x": len(grid_x),
                "grid_size_y": len(grid_y)
            },
            location=np.array([grid_x.mean(), grid_y.mean(), np.nanmean(heights)])
        )

        self.elements.append(element)
        logger.info(f"Added terrain: {name} (GUID: {guid})")
        return guid

    def add_infrastructure_element(
        self,
        name: str,
        element_type: str,
        points: np.ndarray,
        properties: Optional[Dict] = None
    ) -> str:
        """
        Dodaj element infrastruktury

        Args:
            name: nazwa
            element_type: typ (pole, wire, track, etc.)
            points: (N, 3) punkty elementu
            properties: wlasciwosci

        Returns:
            GUID elementu
        """
        guid = self._generate_guid()

        # Bounding box
        bbox = {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist()
        }

        ifc_type = self._map_to_ifc_type(element_type)

        element = BIMElement(
            guid=guid,
            name=name,
            element_type=ifc_type,
            geometry={
                "type": "point_cloud",
                "point_count": len(points),
                "bounds": bbox,
                "centroid": points.mean(axis=0).tolist()
            },
            properties=properties or {},
            classification=element_type,
            location=points.mean(axis=0)
        )

        self.elements.append(element)
        return guid

    def _map_to_ifc_type(self, element_type: str) -> str:
        """Mapuj typ elementu na typ IFC"""
        mapping = {
            "building": "IfcBuilding",
            "terrain": "IfcSite",
            "pole": "IfcColumn",
            "wire": "IfcCableSegment",
            "track": "IfcRail",
            "signal": "IfcSignal",
            "vegetation": "IfcGeographicElement",
            "road": "IfcRoad",
            "bridge": "IfcBridge",
        }
        return mapping.get(element_type.lower(), "IfcBuildingElementProxy")

    def add_point_cloud(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        name: str = "PointCloud",
        sample_rate: float = 0.1
    ) -> str:
        """
        Dodaj chmure punktow (probkowana)

        Args:
            coords: wspolrzedne
            classification: klasyfikacja
            name: nazwa
            sample_rate: procent punktow do eksportu

        Returns:
            GUID elementu
        """
        guid = self._generate_guid()

        # Probkowanie
        n_samples = int(len(coords) * sample_rate)
        indices = np.random.choice(len(coords), n_samples, replace=False)
        sampled_coords = coords[indices]
        sampled_class = classification[indices]

        element = BIMElement(
            guid=guid,
            name=name,
            element_type="IfcPointCloud",
            geometry={
                "type": "point_cloud",
                "point_count": len(sampled_coords),
                "bounds": {
                    "min": sampled_coords.min(axis=0).tolist(),
                    "max": sampled_coords.max(axis=0).tolist()
                }
            },
            properties={
                "original_point_count": len(coords),
                "sampled_point_count": len(sampled_coords),
                "sample_rate": sample_rate,
                "classes": np.unique(sampled_class).tolist()
            },
            location=sampled_coords.mean(axis=0)
        )

        # Zapisz punkty do osobnego atrybutu
        element.geometry["points"] = sampled_coords.tolist()
        element.geometry["classification"] = sampled_class.tolist()

        self.elements.append(element)
        logger.info(f"Added point cloud: {name} ({len(sampled_coords):,} points)")
        return guid

    def export(self, output_path: str, format: str = "ifc") -> bool:
        """
        Eksportuj do pliku

        Args:
            output_path: sciezka wyjsciowa
            format: format (ifc, json, xml)

        Returns:
            True jesli sukces
        """
        try:
            if format.lower() == "ifc":
                return self._export_ifc(output_path)
            elif format.lower() == "json":
                return self._export_json(output_path)
            elif format.lower() == "xml":
                return self._export_xml(output_path)
            else:
                logger.error(f"Unknown format: {format}")
                return False
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _export_ifc(self, output_path: str) -> bool:
        """Eksport do IFC-SPF (STEP Physical File)"""
        logger.info(f"Exporting to IFC: {output_path}")

        # Header
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        header = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('{Path(output_path).name}','{timestamp}',('{self.author}'),('{self.organization}'),'CPK Chmura+ IFC Exporter','{self.IFC_VERSION}','');
FILE_SCHEMA(('{self.IFC_VERSION}'));
ENDSEC;

DATA;
"""

        # Entities
        self._entities = []
        self._entity_id = 1

        # Project
        project_id = self._add_entity("IFCPROJECT", [
            f"'{self._generate_guid()}'",
            "$",
            f"'{self.project_name}'",
            "$", "$", "$", "$", "$", "$"
        ])

        # Site
        site_id = self._add_entity("IFCSITE", [
            f"'{self._generate_guid()}'",
            "$",
            "'Site'",
            "$", "$", "$", "$", "$",
            ".ELEMENT.",
            "$", "$", "$", "$", "$"
        ])

        # Buildings and elements
        for element in self.elements:
            self._add_element_entity(element, site_id)

        # Build content
        content = header
        for entity in self._entities:
            content += f"#{entity['id']}={entity['type']}({','.join(str(p) for p in entity['params'])});\n"

        content += "ENDSEC;\nEND-ISO-10303-21;\n"

        # Write
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"IFC exported: {len(self.elements)} elements")
        return True

    def _add_entity(self, entity_type: str, params: List) -> int:
        """Dodaj encje IFC"""
        entity_id = self._entity_id
        self._entities.append({
            'id': entity_id,
            'type': entity_type,
            'params': params
        })
        self._entity_id += 1
        return entity_id

    def _add_element_entity(self, element: BIMElement, parent_id: int) -> int:
        """Dodaj element jako encje IFC"""
        ifc_type = element.element_type.upper()

        if element.location is not None:
            loc = element.location
            placement_id = self._add_entity("IFCLOCALPLACEMENT", [
                "$",
                f"#{self._add_axis_placement(loc)}"
            ])
        else:
            placement_id = "$"

        params = [
            f"'{element.guid}'",
            "$",
            f"'{element.name}'",
            "$", "$",
            f"#{placement_id}" if isinstance(placement_id, int) else "$",
            "$", "$"
        ]

        # Add type-specific params
        if ifc_type == "IFCBUILDING":
            params.extend(["$", ".ELEMENT.", "$", "$"])
        elif ifc_type == "IFCSITE":
            params.extend([".ELEMENT.", "$", "$", "$", "$", "$"])

        return self._add_entity(ifc_type, params)

    def _add_axis_placement(self, location: np.ndarray) -> int:
        """Dodaj placement 3D"""
        point_id = self._add_entity("IFCCARTESIANPOINT", [
            f"({location[0]:.3f},{location[1]:.3f},{location[2]:.3f})"
        ])
        return self._add_entity("IFCAXIS2PLACEMENT3D", [f"#{point_id}", "$", "$"])

    def _export_json(self, output_path: str) -> bool:
        """Eksport do JSON (BIM-like structure)"""
        logger.info(f"Exporting to JSON: {output_path}")

        data = {
            "project": {
                "name": self.project_name,
                "author": self.author,
                "organization": self.organization,
                "created": datetime.now().isoformat(),
                "ifc_version": self.IFC_VERSION
            },
            "elements": []
        }

        for element in self.elements:
            elem_data = {
                "guid": element.guid,
                "name": element.name,
                "type": element.element_type,
                "classification": element.classification,
                "properties": element.properties,
                "geometry": {
                    k: v for k, v in element.geometry.items()
                    if k != "points"  # Exclude large point arrays
                }
            }
            if element.location is not None:
                elem_data["location"] = element.location.tolist()

            data["elements"].append(elem_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON exported: {len(self.elements)} elements")
        return True

    def _export_xml(self, output_path: str) -> bool:
        """Eksport do XML (BIM-like structure)"""
        logger.info(f"Exporting to XML: {output_path}")

        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<BIMProject xmlns="http://cpk.chmuraplus.pl/bim">
    <Header>
        <Name>{self.project_name}</Name>
        <Author>{self.author}</Author>
        <Organization>{self.organization}</Organization>
        <Created>{datetime.now().isoformat()}</Created>
        <IFCVersion>{self.IFC_VERSION}</IFCVersion>
    </Header>
    <Elements>
"""

        for element in self.elements:
            xml_content += f"""        <Element>
            <GUID>{element.guid}</GUID>
            <Name>{element.name}</Name>
            <Type>{element.element_type}</Type>
"""
            if element.location is not None:
                xml_content += f"""            <Location>
                <X>{element.location[0]:.3f}</X>
                <Y>{element.location[1]:.3f}</Y>
                <Z>{element.location[2]:.3f}</Z>
            </Location>
"""
            # Properties
            if element.properties:
                xml_content += "            <Properties>\n"
                for key, value in element.properties.items():
                    xml_content += f"                <{key}>{value}</{key}>\n"
                xml_content += "            </Properties>\n"

            xml_content += "        </Element>\n"

        xml_content += """    </Elements>
</BIMProject>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)

        logger.info(f"XML exported: {len(self.elements)} elements")
        return True

    def get_summary(self) -> Dict:
        """Podsumowanie elementow do eksportu"""
        type_counts = {}
        for element in self.elements:
            t = element.element_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "project_name": self.project_name,
            "total_elements": len(self.elements),
            "elements_by_type": type_counts,
            "ifc_version": self.IFC_VERSION
        }


def export_to_ifc(
    buildings: List[Any],
    terrain: Optional[Any] = None,
    output_path: str = "export.ifc",
    project_name: str = "CPK Project"
) -> bool:
    """
    Convenience function - eksportuj do IFC

    Args:
        buildings: lista budynkow (z BuildingExtractor)
        terrain: model terenu (TerrainModel)
        output_path: sciezka wyjsciowa
        project_name: nazwa projektu

    Returns:
        True jesli sukces
    """
    exporter = IFCExporter(project_name=project_name)

    # Dodaj budynki
    for i, building in enumerate(buildings):
        if hasattr(building, 'footprint'):
            exporter.add_building(
                name=f"Building_{i+1}",
                footprint_vertices=building.footprint.vertices,
                height=building.height,
                ground_elevation=building.height_min,
                properties={
                    "roof_type": building.roof_type.value if hasattr(building.roof_type, 'value') else str(building.roof_type),
                    "floor_count": building.floor_count_estimate,
                    "area_m2": building.footprint.area_m2,
                    "volume_m3": building.volume_m3
                }
            )

    # Dodaj teren
    if terrain is not None and hasattr(terrain, 'grid_x'):
        exporter.add_terrain(
            grid_x=terrain.grid_x,
            grid_y=terrain.grid_y,
            heights=terrain.heights
        )

    return exporter.export(output_path)
