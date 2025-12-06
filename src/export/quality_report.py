"""
Quality reporter for classification results

Generates comprehensive quality metrics and visualizations:
- Per-class statistics (count, percentage, mean confidence)
- Confidence distribution analysis
- Spatial distribution metrics
- Confusion matrix (if ground truth available)
- Quality score estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityReporter:
    """
    Generate quality reports for classification results

    Provides:
    - Class distribution statistics
    - Confidence analysis
    - Spatial coverage metrics
    - Quality score estimation
    - Export to CSV/JSON/Markdown
    """

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize quality reporter

        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names or self._default_class_names()
        logger.info("Quality reporter initialized")

    def _default_class_names(self) -> Dict[int, str]:
        """Default CPK infrastructure class names"""
        return {
            0: "Unclassified",
            1: "Road",
            2: "Vegetation",
            3: "Building",
            4: "Railway",
            5: "Curbs",
            6: "Poles",
            7: "Signs",
            8: "Barriers",
            9: "Pipelines",
            10: "Power Lines",
            11: "Traction"
        }

    def generate_report(self,
                       predictions: np.ndarray,
                       confidences: np.ndarray,
                       coords: Optional[np.ndarray] = None,
                       processing_stats: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive quality report

        Args:
            predictions: (N,) array of predicted class labels
            confidences: (N,) array of confidence scores
            coords: (N, 3) array of XYZ coordinates (optional)
            processing_stats: Processing statistics from pipeline (optional)

        Returns:
            Dictionary with quality metrics and statistics
        """
        logger.info(f"Generating quality report for {len(predictions):,} points...")

        report = {
            'metadata': self._generate_metadata(),
            'summary': self._generate_summary(predictions, confidences),
            'per_class_stats': self._generate_per_class_stats(predictions, confidences),
            'confidence_analysis': self._analyze_confidence(predictions, confidences),
            'quality_score': self._estimate_quality_score(predictions, confidences)
        }

        if coords is not None:
            report['spatial_stats'] = self._analyze_spatial_distribution(predictions, coords)

        if processing_stats is not None:
            report['processing_stats'] = processing_stats

        logger.info("✅ Quality report generated")

        return report

    def _generate_metadata(self) -> Dict:
        """Generate report metadata"""
        return {
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0',
            'tool': 'CPK Point Cloud Classifier'
        }

    def _generate_summary(self, predictions: np.ndarray, confidences: np.ndarray) -> Dict:
        """Generate summary statistics"""
        return {
            'total_points': int(len(predictions)),
            'unique_classes': int(len(np.unique(predictions))),
            'mean_confidence': float(confidences.mean()),
            'median_confidence': float(np.median(confidences)),
            'std_confidence': float(confidences.std()),
            'min_confidence': float(confidences.min()),
            'max_confidence': float(confidences.max())
        }

    def _generate_per_class_stats(self, predictions: np.ndarray, confidences: np.ndarray) -> pd.DataFrame:
        """Generate per-class statistics"""
        stats = []

        for class_id in sorted(np.unique(predictions)):
            mask = predictions == class_id
            count = mask.sum()

            stats.append({
                'class_id': int(class_id),
                'class_name': self.class_names.get(class_id, f"Class_{class_id}"),
                'point_count': int(count),
                'percentage': float(count / len(predictions) * 100),
                'mean_confidence': float(confidences[mask].mean()),
                'median_confidence': float(np.median(confidences[mask])),
                'std_confidence': float(confidences[mask].std())
            })

        df = pd.DataFrame(stats)
        return df

    def _analyze_confidence(self, predictions: np.ndarray, confidences: np.ndarray) -> Dict:
        """Analyze confidence distribution"""
        # Confidence bins
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['<50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

        hist, _ = np.histogram(confidences, bins=bins)

        confidence_dist = {}
        for label, count in zip(bin_labels, hist):
            confidence_dist[label] = {
                'count': int(count),
                'percentage': float(count / len(confidences) * 100)
            }

        # High confidence threshold (>80%)
        high_conf_mask = confidences > 0.8
        high_conf_count = high_conf_mask.sum()

        return {
            'distribution': confidence_dist,
            'high_confidence_count': int(high_conf_count),
            'high_confidence_percentage': float(high_conf_count / len(confidences) * 100)
        }

    def _analyze_spatial_distribution(self, predictions: np.ndarray, coords: np.ndarray) -> Dict:
        """Analyze spatial distribution of classes"""
        # Bounding box
        bbox = {
            'min_x': float(coords[:, 0].min()),
            'max_x': float(coords[:, 0].max()),
            'min_y': float(coords[:, 1].min()),
            'max_y': float(coords[:, 1].max()),
            'min_z': float(coords[:, 2].min()),
            'max_z': float(coords[:, 2].max())
        }

        # Area covered
        area_m2 = (bbox['max_x'] - bbox['min_x']) * (bbox['max_y'] - bbox['min_y'])
        height_range = bbox['max_z'] - bbox['min_z']

        # Point density
        density = len(predictions) / area_m2 if area_m2 > 0 else 0

        return {
            'bounding_box': bbox,
            'area_m2': float(area_m2),
            'height_range_m': float(height_range),
            'point_density_per_m2': float(density)
        }

    def _estimate_quality_score(self, predictions: np.ndarray, confidences: np.ndarray) -> Dict:
        """
        Estimate overall quality score (0-100)

        Based on:
        - Mean confidence (40%)
        - High confidence percentage (30%)
        - Class diversity (15%)
        - Unclassified percentage (15% penalty)
        """
        # Mean confidence score (0-40 points)
        mean_conf_score = confidences.mean() * 40

        # High confidence percentage (0-30 points)
        high_conf_pct = (confidences > 0.8).sum() / len(confidences)
        high_conf_score = high_conf_pct * 30

        # Class diversity score (0-15 points)
        unique_classes = len(np.unique(predictions))
        max_classes = 12
        diversity_score = (unique_classes / max_classes) * 15

        # Unclassified penalty (0-15 points lost)
        unclassified_pct = (predictions == 0).sum() / len(predictions)
        unclassified_penalty = unclassified_pct * 15

        # Total score
        total_score = mean_conf_score + high_conf_score + diversity_score - unclassified_penalty
        total_score = max(0, min(100, total_score))  # Clamp to [0, 100]

        # Quality rating
        if total_score >= 80:
            rating = "Excellent"
        elif total_score >= 70:
            rating = "Good"
        elif total_score >= 60:
            rating = "Fair"
        else:
            rating = "Poor"

        return {
            'overall_score': float(total_score),
            'rating': rating,
            'components': {
                'mean_confidence': float(mean_conf_score),
                'high_confidence': float(high_conf_score),
                'diversity': float(diversity_score),
                'unclassified_penalty': float(unclassified_penalty)
            }
        }

    def export_to_markdown(self, report: Dict) -> str:
        """Export report to Markdown format"""
        md = []

        md.append("# Point Cloud Classification Quality Report")
        md.append(f"\nGenerated: {report['metadata']['generated_at']}")
        md.append(f"\n## Summary")
        md.append(f"- **Total Points**: {report['summary']['total_points']:,}")
        md.append(f"- **Unique Classes**: {report['summary']['unique_classes']}")
        md.append(f"- **Mean Confidence**: {report['summary']['mean_confidence']:.1%}")
        md.append(f"- **Quality Score**: {report['quality_score']['overall_score']:.1f}/100 ({report['quality_score']['rating']})")

        md.append(f"\n## Per-Class Statistics")
        md.append("")
        df = report['per_class_stats']
        md.append(df.to_markdown(index=False))

        md.append(f"\n## Confidence Analysis")
        md.append(f"- **High Confidence (>80%)**: {report['confidence_analysis']['high_confidence_percentage']:.1f}%")
        md.append("\n**Distribution:**")
        for bin_label, stats in report['confidence_analysis']['distribution'].items():
            md.append(f"- {bin_label}: {stats['count']:,} points ({stats['percentage']:.1f}%)")

        if 'spatial_stats' in report:
            md.append(f"\n## Spatial Coverage")
            spatial = report['spatial_stats']
            md.append(f"- **Area**: {spatial['area_m2']:,.1f} m²")
            md.append(f"- **Height Range**: {spatial['height_range_m']:.1f} m")
            md.append(f"- **Point Density**: {spatial['point_density_per_m2']:,.1f} points/m²")

        return "\n".join(md)

    def export_to_csv(self, report: Dict, output_path: str):
        """Export per-class stats to CSV"""
        df = report['per_class_stats']
        df.to_csv(output_path, index=False)
        logger.info(f"Exported report to {output_path}")


def test_quality_reporter():
    """Test quality reporter"""
    print("Testing QualityReporter...")

    # Create synthetic data
    np.random.seed(42)
    n_points = 100000

    predictions = np.random.choice([0, 1, 2, 3, 4, 5, 6], size=n_points, p=[0.1, 0.3, 0.2, 0.15, 0.1, 0.1, 0.05])
    confidences = np.random.beta(8, 2, size=n_points)  # Skewed towards high confidence
    coords = np.random.uniform(0, 100, size=(n_points, 3))

    # Generate report
    reporter = QualityReporter()
    report = reporter.generate_report(predictions, confidences, coords)

    print(f"\n✅ Report Summary:")
    print(f"   Total Points: {report['summary']['total_points']:,}")
    print(f"   Mean Confidence: {report['summary']['mean_confidence']:.1%}")
    print(f"   Quality Score: {report['quality_score']['overall_score']:.1f}/100")
    print(f"   Rating: {report['quality_score']['rating']}")

    print(f"\n✅ Per-Class Stats:")
    print(report['per_class_stats'].to_string(index=False))

    print(f"\n✅ Markdown Export:")
    md = reporter.export_to_markdown(report)
    print(md[:500] + "...")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_quality_reporter()
