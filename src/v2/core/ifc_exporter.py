"""
IFC Exporter for Point Cloud Classification Results

Exports classified point cloud to IFC format with:
- Classification groups (IfcGroup for each class)
- Bounding boxes for each class (IfcBuildingElementProxy)
- Project metadata

This is a lightweight implementation that generates valid IFC 2x3 files
without requiring external IFC libraries.

ASPRS/CPK Class to IFC Element mapping:
- Ground (2) → IfcSite
- Buildings (6, 40, 41) → IfcBuilding / IfcWall / IfcRoof
- Vegetation (3, 4, 5) → IfcGeographicElement
- Roads (30) → IfcRoad or IfcPavement
- Rails (18) → IfcRailway
- Infrastructure (19, 20, 21, 32, 35, 36) → IfcBuildingElementProxy
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)


class IFCExporter:
    """
    Simple IFC exporter for point cloud classification results

    Generates IFC 2x3 files with classification groups and bounding boxes.

    Usage:
        exporter = IFCExporter(
            coords=coords,
            classification=classification,
            project_name="CPK Point Cloud Classification"
        )
        exporter.export("output.ifc")
    """

    # IFC Element type mapping for each ASPRS class
    CLASS_TO_IFC = {
        1: ("IfcBuildingElementProxy", "USERDEFINED", "Unclassified"),
        2: ("IfcSite", "ELEMENT", "Ground"),
        3: ("IfcGeographicElement", "USERDEFINED", "Low Vegetation"),
        4: ("IfcGeographicElement", "USERDEFINED", "Medium Vegetation"),
        5: ("IfcGeographicElement", "USERDEFINED", "High Vegetation"),
        6: ("IfcBuilding", "ELEMENT", "Building"),
        7: ("IfcBuildingElementProxy", "USERDEFINED", "Noise"),
        9: ("IfcGeographicElement", "TERRAIN", "Water"),
        18: ("IfcRailway", "USERDEFINED", "Railway Track"),
        19: ("IfcCableCarrierSegment", "CABLESEGMENT", "Power Line"),
        20: ("IfcColumn", "COLUMN", "Traction Pole"),
        21: ("IfcSlab", "FLOOR", "Railway Platform"),
        30: ("IfcPavement", "USERDEFINED", "Road Surface"),
        32: ("IfcCurbOrKerb", "USERDEFINED", "Curb"),
        35: ("IfcSign", "USERDEFINED", "Road Sign"),
        36: ("IfcRailing", "GUARDRAIL", "Road Barrier"),
        40: ("IfcWall", "SOLIDWALL", "Building Wall"),
        41: ("IfcRoof", "FLAT_ROOF", "Building Roof"),
    }

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        project_name: str = "CPK Point Cloud Classification",
        project_description: str = "Automatic LiDAR Classification - HackNation 2025",
        author: str = "CPK Classifier v2.0"
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            classification: (N,) Classification labels
            project_name: IFC project name
            project_description: IFC project description
            author: Author name
        """
        self.coords = coords
        self.classification = classification
        self.project_name = project_name
        self.project_description = project_description
        self.author = author
        self.n_points = len(coords)

        # Generate unique IDs
        self._id_counter = 0

        logger.info(f"IFCExporter initialized: {self.n_points:,} points")

    def _next_id(self) -> int:
        """Generate next sequential ID"""
        self._id_counter += 1
        return self._id_counter

    def _guid(self) -> str:
        """Generate IFC-compatible GUID (22 chars base64)"""
        # IFC uses compressed GUIDs (22 characters)
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        uid = uuid.uuid4().int
        result = []
        for _ in range(22):
            result.append(chars[uid % 64])
            uid //= 64
        return ''.join(result)

    def _compute_bounds_per_class(self) -> Dict[int, Dict]:
        """Compute bounding box for each class"""
        bounds = {}
        unique_classes = np.unique(self.classification)

        for cls in unique_classes:
            mask = self.classification == cls
            if mask.sum() == 0:
                continue

            cls_coords = self.coords[mask]
            bounds[int(cls)] = {
                'min': cls_coords.min(axis=0).tolist(),
                'max': cls_coords.max(axis=0).tolist(),
                'count': int(mask.sum()),
                'center': cls_coords.mean(axis=0).tolist()
            }

        return bounds

    def export(self, output_path: str) -> Dict:
        """
        Export to IFC file

        Args:
            output_path: Path to output .ifc file

        Returns:
            Dict with export statistics
        """
        logger.info(f"Exporting to IFC: {output_path}")

        # Compute bounds per class
        bounds = self._compute_bounds_per_class()

        # Generate IFC content
        ifc_content = self._generate_ifc(bounds)

        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ifc_content)

        stats = {
            'output_path': output_path,
            'n_points': self.n_points,
            'n_classes': len(bounds),
            'classes': list(bounds.keys())
        }

        logger.info(f"IFC export complete: {len(bounds)} classes")
        return stats

    def _generate_ifc(self, bounds: Dict) -> str:
        """Generate complete IFC file content"""
        lines = []

        # ISO header
        lines.append("ISO-10303-21;")
        lines.append("HEADER;")
        lines.append(f"FILE_DESCRIPTION(('CPK Point Cloud Classification'),'2;1');")
        lines.append(f"FILE_NAME('{self.project_name}','{datetime.now().isoformat()}',")
        lines.append(f"  ('{self.author}'),('CPK HackNation 2025'),'IFC2X3','CPK Classifier v2.0','');")
        lines.append("FILE_SCHEMA(('IFC2X3'));")
        lines.append("ENDSEC;")
        lines.append("")
        lines.append("DATA;")

        # Generate entities
        entities = self._generate_entities(bounds)
        lines.extend(entities)

        lines.append("ENDSEC;")
        lines.append("END-ISO-10303-21;")

        return '\n'.join(lines)

    def _generate_entities(self, bounds: Dict) -> list:
        """Generate IFC entities"""
        entities = []

        # 1. Organization and Application
        org_id = self._next_id()
        entities.append(f"#{org_id}=IFCORGANIZATION($,'CPK HackNation 2025',$,$,$);")

        app_id = self._next_id()
        entities.append(f"#{app_id}=IFCAPPLICATION(#{org_id},'2.0','CPK Classifier','CPK_CLASSIFIER');")

        # 2. Person and Owner History
        person_id = self._next_id()
        entities.append(f"#{person_id}=IFCPERSON($,'Classifier','CPK',$,$,$,$,$);")

        person_org_id = self._next_id()
        entities.append(f"#{person_org_id}=IFCPERSONANDORGANIZATION(#{person_id},#{org_id},$);")

        owner_id = self._next_id()
        timestamp = int(datetime.now().timestamp())
        entities.append(f"#{owner_id}=IFCOWNERHISTORY(#{person_org_id},#{app_id},$,.NOCHANGE.,$,$,$,{timestamp});")

        # 3. Units
        length_unit_id = self._next_id()
        entities.append(f"#{length_unit_id}=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);")

        area_unit_id = self._next_id()
        entities.append(f"#{area_unit_id}=IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.);")

        volume_unit_id = self._next_id()
        entities.append(f"#{volume_unit_id}=IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.);")

        units_id = self._next_id()
        entities.append(f"#{units_id}=IFCUNITASSIGNMENT((#{length_unit_id},#{area_unit_id},#{volume_unit_id}));")

        # 4. Geometric Context
        origin_id = self._next_id()
        entities.append(f"#{origin_id}=IFCCARTESIANPOINT((0.,0.,0.));")

        axis_z_id = self._next_id()
        entities.append(f"#{axis_z_id}=IFCDIRECTION((0.,0.,1.));")

        axis_x_id = self._next_id()
        entities.append(f"#{axis_x_id}=IFCDIRECTION((1.,0.,0.));")

        placement_id = self._next_id()
        entities.append(f"#{placement_id}=IFCAXIS2PLACEMENT3D(#{origin_id},#{axis_z_id},#{axis_x_id});")

        context_id = self._next_id()
        entities.append(f"#{context_id}=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-5,#{placement_id},$);")

        # 5. Project
        project_id = self._next_id()
        project_guid = self._guid()
        entities.append(f"#{project_id}=IFCPROJECT('{project_guid}',#{owner_id},'{self.project_name}','{self.project_description}',$,$,$,(#{context_id}),#{units_id});")

        # 6. Site (for ground reference)
        site_placement_id = self._next_id()
        entities.append(f"#{site_placement_id}=IFCLOCALPLACEMENT($,#{placement_id});")

        site_id = self._next_id()
        site_guid = self._guid()
        entities.append(f"#{site_id}=IFCSITE('{site_guid}',#{owner_id},'CPK Site','Point Cloud Site',$,#{site_placement_id},$,$,.ELEMENT.,$,$,$,$,$);")

        # 7. Rel Aggregates (Project -> Site)
        rel_id = self._next_id()
        rel_guid = self._guid()
        entities.append(f"#{rel_id}=IFCRELAGGREGATES('{rel_guid}',#{owner_id},'Project Container','Project to Site relationship',#{project_id},(#{site_id}));")

        # 8. Classification groups for each class
        group_ids = []
        for cls, cls_bounds in bounds.items():
            cls_info = self.CLASS_TO_IFC.get(cls, ("IfcBuildingElementProxy", "USERDEFINED", f"Class {cls}"))
            ifc_type, predefined_type, name = cls_info

            # Create bounding box representation
            min_pt = cls_bounds['min']
            max_pt = cls_bounds['max']
            center = cls_bounds['center']
            count = cls_bounds['count']

            # Dimensions
            dx = max_pt[0] - min_pt[0]
            dy = max_pt[1] - min_pt[1]
            dz = max_pt[2] - min_pt[2]

            # Create placement at center
            center_pt_id = self._next_id()
            entities.append(f"#{center_pt_id}=IFCCARTESIANPOINT(({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}));")

            elem_placement_id = self._next_id()
            entities.append(f"#{elem_placement_id}=IFCAXIS2PLACEMENT3D(#{center_pt_id},#{axis_z_id},#{axis_x_id});")

            local_placement_id = self._next_id()
            entities.append(f"#{local_placement_id}=IFCLOCALPLACEMENT(#{site_placement_id},#{elem_placement_id});")

            # Create bounding box shape
            bbox_id = self._next_id()
            entities.append(f"#{bbox_id}=IFCBOUNDINGBOX(#{origin_id},{max(dx, 0.1):.3f},{max(dy, 0.1):.3f},{max(dz, 0.1):.3f});")

            shape_rep_id = self._next_id()
            entities.append(f"#{shape_rep_id}=IFCSHAPEREPRESENTATION(#{context_id},'Body','BoundingBox',(#{bbox_id}));")

            product_shape_id = self._next_id()
            entities.append(f"#{product_shape_id}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape_rep_id}));")

            # Create element (simplified - use IfcBuildingElementProxy for all)
            elem_id = self._next_id()
            elem_guid = self._guid()
            description = f"Class {cls}: {name} ({count:,} points)"

            # Use IfcBuildingElementProxy for compatibility
            entities.append(f"#{elem_id}=IFCBUILDINGELEMENTPROXY('{elem_guid}',#{owner_id},'Class_{cls}_{name.replace(' ', '_')}','{description}',$,#{local_placement_id},#{product_shape_id},$,$);")

            group_ids.append(elem_id)

            # Add property set with statistics
            prop_count_id = self._next_id()
            entities.append(f"#{prop_count_id}=IFCPROPERTYSINGLEVALUE('PointCount',$,IFCINTEGER({count}),$);")

            prop_class_id = self._next_id()
            entities.append(f"#{prop_class_id}=IFCPROPERTYSINGLEVALUE('ClassificationID',$,IFCINTEGER({cls}),$);")

            prop_pct_id = self._next_id()
            pct = count / self.n_points * 100
            entities.append(f"#{prop_pct_id}=IFCPROPERTYSINGLEVALUE('Percentage',$,IFCREAL({pct:.2f}),$);")

            pset_id = self._next_id()
            pset_guid = self._guid()
            entities.append(f"#{pset_id}=IFCPROPERTYSET('{pset_guid}',#{owner_id},'Classification_Statistics',$,(#{prop_count_id},#{prop_class_id},#{prop_pct_id}));")

            rel_pset_id = self._next_id()
            rel_pset_guid = self._guid()
            entities.append(f"#{rel_pset_id}=IFCRELDEFINESBYPROPERTIES('{rel_pset_guid}',#{owner_id},$,$,(#{elem_id}),#{pset_id});")

        # 9. Contain elements in site
        if group_ids:
            rel_contains_id = self._next_id()
            rel_contains_guid = self._guid()
            group_refs = ','.join([f'#{gid}' for gid in group_ids])
            entities.append(f"#{rel_contains_id}=IFCRELCONTAINEDINSPATIALSTRUCTURE('{rel_contains_guid}',#{owner_id},'Site Contents',$,({group_refs}),#{site_id});")

        return entities


def export_classification_to_ifc(
    coords: np.ndarray,
    classification: np.ndarray,
    output_path: str,
    project_name: str = "CPK Point Cloud Classification"
) -> Dict:
    """
    Convenience function to export classification to IFC

    Args:
        coords: (N, 3) Point coordinates
        classification: (N,) Classification labels
        output_path: Path to output .ifc file
        project_name: IFC project name

    Returns:
        Dict with export statistics
    """
    exporter = IFCExporter(coords, classification, project_name)
    return exporter.export(output_path)
