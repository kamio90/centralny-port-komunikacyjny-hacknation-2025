"""
Integration tests for complete classification pipeline

Tests the full workflow:
1. Load LAS file
2. Create tiles
3. Extract features
4. Run ML inference
5. Apply post-processing
6. Generate quality report
7. Export results
"""

import numpy as np
import torch
from pathlib import Path
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_extraction():
    """Test feature extractor"""
    logger.info("Testing feature extraction...")

    from src.data.features import FeatureExtractor

    # Create synthetic point cloud
    n_points = 1000
    coords = np.random.uniform(0, 100, size=(n_points, 3)).astype(np.float32)
    colors = np.random.uniform(0, 1, size=(n_points, 3)).astype(np.float32)
    intensity = np.random.uniform(0, 1, size=n_points).astype(np.float32)

    # Extract features
    extractor = FeatureExtractor(normalize=True)
    features = extractor.extract(coords, colors, intensity)

    # Verify output
    assert features.shape == (n_points, 9), f"Expected (1000, 9), got {features.shape}"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    assert np.all(np.isfinite(features)), "Features contain NaN or Inf"

    logger.info("✅ Feature extraction test passed")
    return True


def test_model_inference():
    """Test ML model inference"""
    logger.info("Testing ML model inference...")

    from src.models.pointnet2_lite import PointNet2Lite
    from src.models.inference import InferenceEngine

    # Create model
    model = PointNet2Lite(num_classes=12, input_channels=9)
    model.eval()

    # Create synthetic features
    batch_size = 2
    n_points = 1000
    features = np.random.randn(batch_size * n_points, 9).astype(np.float32)

    # Create inference engine
    engine = InferenceEngine(model, device='cpu', batch_size=1000)

    # Run inference
    predictions, confidences = engine.predict(features)

    # Verify output
    assert predictions.shape == (batch_size * n_points,), f"Unexpected predictions shape: {predictions.shape}"
    assert confidences.shape == (batch_size * n_points,), f"Unexpected confidences shape: {confidences.shape}"
    assert np.issubdtype(predictions.dtype, np.integer), f"Expected integer dtype, got {predictions.dtype}"
    assert np.all((predictions >= 0) & (predictions < 12)), "Invalid prediction values"
    assert np.all((confidences >= 0) & (confidences <= 1)), "Confidence values out of range"

    logger.info("✅ Model inference test passed")
    return True


def test_postprocessing():
    """Test post-processing filters"""
    logger.info("Testing post-processing...")

    from src.postprocessing.confidence_filter import ConfidenceFilter
    from src.postprocessing.dbscan_cleaner import DBSCANCleaner

    # Create synthetic data
    n_points = 10000
    predictions = np.random.randint(0, 12, size=n_points)
    confidences = np.random.uniform(0.4, 1.0, size=n_points)
    coords = np.random.uniform(0, 100, size=(n_points, 3))

    # Test confidence filter
    conf_filter = ConfidenceFilter(global_threshold=0.7)
    filtered_preds, stats = conf_filter.filter(predictions, confidences)

    assert filtered_preds.shape == predictions.shape, "Shape mismatch after filtering"
    assert stats['reassigned_count'] > 0, "Expected some reassignments"
    assert stats['reassignment_rate'] > 0, "Expected non-zero reassignment rate"

    # Test DBSCAN cleaner
    # Create clustered data for class 6 (poles)
    cluster_coords = np.random.randn(500, 3) * 0.5 + np.array([50, 50, 2])
    cluster_preds = np.full(500, 6)

    # Add some noise points
    noise_coords = np.random.uniform(0, 100, size=(50, 3))
    noise_preds = np.full(50, 6)

    test_coords = np.vstack([cluster_coords, noise_coords])
    test_preds = np.concatenate([cluster_preds, noise_preds])

    cleaner = DBSCANCleaner()
    cleaned_preds, dbscan_stats = cleaner.clean(test_preds, test_coords)

    assert cleaned_preds.shape == test_preds.shape, "Shape mismatch after DBSCAN"
    assert dbscan_stats['total_removed'] >= 0, "Invalid removal count"

    logger.info("✅ Post-processing test passed")
    return True


def test_export_las():
    """Test LAS/LAZ export"""
    logger.info("Testing LAS export...")

    from src.export.las_writer import LASWriter
    import laspy

    # Create synthetic data
    n_points = 1000
    coords = np.random.uniform(0, 100, size=(n_points, 3))
    classifications = np.random.randint(0, 12, size=n_points)
    colors = np.random.uniform(0, 1, size=(n_points, 3))
    confidences = np.random.uniform(0.5, 1.0, size=n_points)

    # Write LAS file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/test_output.laz"

        writer = LASWriter(output_path, compress=True)
        stats = writer.write(
            coords=coords,
            classifications=classifications,
            colors=colors,
            confidences=confidences
        )

        # Verify file exists
        assert Path(output_path).exists(), "Output file not created"
        assert stats['point_count'] == n_points, "Point count mismatch"

        # Read back and verify
        las = laspy.read(output_path)
        assert len(las.points) == n_points, "Points lost in write/read cycle"
        assert np.allclose(las.x, coords[:, 0], atol=0.01), "X coordinates mismatch"
        assert np.array_equal(las.classification, classifications), "Classifications mismatch"

    logger.info("✅ LAS export test passed")
    return True


def test_quality_report():
    """Test quality report generation"""
    logger.info("Testing quality report...")

    from src.export.quality_report import QualityReporter

    # Create synthetic data
    n_points = 10000
    predictions = np.random.choice([0, 1, 2, 3, 4], size=n_points, p=[0.1, 0.3, 0.3, 0.2, 0.1])
    confidences = np.random.beta(8, 2, size=n_points)
    coords = np.random.uniform(0, 100, size=(n_points, 3))

    # Generate report
    reporter = QualityReporter()
    report = reporter.generate_report(predictions, confidences, coords)

    # Verify report structure
    assert 'metadata' in report, "Missing metadata"
    assert 'summary' in report, "Missing summary"
    assert 'per_class_stats' in report, "Missing per-class stats"
    assert 'confidence_analysis' in report, "Missing confidence analysis"
    assert 'quality_score' in report, "Missing quality score"
    assert 'spatial_stats' in report, "Missing spatial stats"

    # Verify summary
    assert report['summary']['total_points'] == n_points, "Total points mismatch"
    assert 0 <= report['summary']['mean_confidence'] <= 1, "Invalid mean confidence"

    # Verify quality score
    assert 0 <= report['quality_score']['overall_score'] <= 100, "Quality score out of range"
    assert report['quality_score']['rating'] in ['Excellent', 'Good', 'Fair', 'Poor'], "Invalid rating"

    # Test markdown export
    md_report = reporter.export_to_markdown(report)
    assert len(md_report) > 0, "Empty markdown report"
    assert "# Point Cloud Classification Quality Report" in md_report, "Missing title"

    logger.info("✅ Quality report test passed")
    return True


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    logger.info("Testing end-to-end workflow...")

    from src.data.features import FeatureExtractor
    from src.models.pointnet2_lite import PointNet2Lite
    from src.models.inference import InferenceEngine
    from src.postprocessing.confidence_filter import ConfidenceFilter
    from src.postprocessing.dbscan_cleaner import DBSCANCleaner
    from src.export.las_writer import LASWriter
    from src.export.quality_report import QualityReporter

    # Step 1: Create synthetic point cloud
    n_points = 5000
    coords = np.random.uniform(0, 100, size=(n_points, 3)).astype(np.float32)
    colors = np.random.uniform(0, 1, size=(n_points, 3)).astype(np.float32)
    intensity = np.random.uniform(0, 1, size=n_points).astype(np.float32)

    # Step 2: Extract features
    extractor = FeatureExtractor(normalize=True)
    features = extractor.extract(coords, colors, intensity)
    assert features.shape == (n_points, 9), "Feature extraction failed"

    # Step 3: Run ML inference
    model = PointNet2Lite(num_classes=12, input_channels=9)
    model.eval()
    engine = InferenceEngine(model, device='cpu', batch_size=1000)
    predictions, confidences = engine.predict(features)
    assert len(predictions) == n_points, "Inference failed"

    # Step 4: Apply post-processing
    conf_filter = ConfidenceFilter(global_threshold=0.7)
    predictions, _ = conf_filter.filter(predictions, confidences)

    cleaner = DBSCANCleaner()
    predictions, _ = cleaner.clean(predictions, coords)

    # Step 5: Generate quality report
    reporter = QualityReporter()
    report = reporter.generate_report(predictions, confidences, coords)
    assert report['summary']['total_points'] == n_points, "Quality report failed"

    # Step 6: Export results
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/final_output.laz"
        writer = LASWriter(output_path, compress=True)
        stats = writer.write(coords, predictions, colors, confidences=confidences)
        assert Path(output_path).exists(), "Export failed"

    logger.info("✅ End-to-end workflow test passed")
    return True


def run_all_tests():
    """Run all integration tests"""
    logger.info("=" * 60)
    logger.info("Running CPK Point Cloud Classifier Integration Tests")
    logger.info("=" * 60)

    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Model Inference", test_model_inference),
        ("Post-Processing", test_postprocessing),
        ("LAS Export", test_export_las),
        ("Quality Report", test_quality_report),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'─' * 60}")

        try:
            result = test_func()
            results.append((test_name, "PASSED", None))
        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            results.append((test_name, "FAILED", str(e)))

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Test Summary")
    logger.info(f"{'=' * 60}")

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    for test_name, status, error in results:
        symbol = "✅" if status == "PASSED" else "❌"
        logger.info(f"{symbol} {test_name}: {status}")
        if error:
            logger.info(f"   Error: {error}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Total: {len(results)} tests | Passed: {passed} | Failed: {failed}")
    logger.info(f"{'=' * 60}")

    if failed == 0:
        logger.info("\n🎉 All tests passed!")
        return True
    else:
        logger.error(f"\n💥 {failed} test(s) failed")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
