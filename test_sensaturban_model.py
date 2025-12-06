"""
Test script to verify SensatUrban model works correctly
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.las_reader import LASReader
from src.data.features import FeatureExtractor
from src.models.inference import InferenceEngine
from src.models.class_mapper import CPK_CLASSES, print_mapping_table

def test_sensaturban_model():
    """Test SensatUrban model on sample data"""

    print("="*80)
    print("🧪 TEST: SensatUrban Pre-trained Model")
    print("="*80)

    # Print class mapping
    print_mapping_table()

    # Load reference file
    input_file = "hackaton_task_files/Chmura_Zadanie_Dane.las"

    if not Path(input_file).exists():
        print(f"\n❌ File not found: {input_file}")
        print("Please ensure the reference LAS file is available")
        return False

    reader = LASReader(Path(input_file))

    # Sample 100K points
    print("\n📊 Sampling 100,000 points for testing...")
    sample_data = reader.sample_points(n_samples=100_000, method='random')

    print(f"✅ Sampled {sample_data['point_count']:,} points")
    print(f"   Bounds: X [{sample_data['coords'][:, 0].min():.1f}, {sample_data['coords'][:, 0].max():.1f}]")
    print(f"           Y [{sample_data['coords'][:, 1].min():.1f}, {sample_data['coords'][:, 1].max():.1f}]")
    print(f"           Z [{sample_data['coords'][:, 2].min():.1f}, {sample_data['coords'][:, 2].max():.1f}]")

    # Extract features
    print("\n🔧 Extracting features...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract(
        sample_data['coords'],
        sample_data['colors'],
        sample_data['intensity']
    )

    print(f"✅ Features shape: {features.shape}")
    print(f"   Features mean: {features.mean():.3f}")
    print(f"   Features std: {features.std():.3f}")

    # Load model
    print("\n📦 Loading SensatUrban model...")
    try:
        inference_engine = InferenceEngine(
            model=None,  # Will load SensatUrban
            device='auto',
            batch_size=50000,
            use_pretrained=True
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("1. Check if models/pretrained/randlanet_sensaturban.pth exists")
        print("2. If not, download it manually from:")
        print("   https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_sensaturban.pth")
        return False

    # Predict
    print("\n🚀 Running inference...")
    predictions, confidences = inference_engine.predict(features)

    # Analyze results
    print("\n" + "="*80)
    print("📊 WYNIKI KLASYFIKACJI (CLASSIFICATION RESULTS)")
    print("="*80)

    unique, counts = np.unique(predictions, return_counts=True)

    results = []
    for class_id, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        class_name = CPK_CLASSES[class_id]
        results.append((class_name, count, percentage))
        print(f"{class_name:30s}: {count:8,} punktów ({percentage:5.1f}%)")

    print("="*80)
    print(f"Średnia pewność: {confidences.mean():.2%}")
    print(f"Mediana pewności: {np.median(confidences):.2%}")
    print(f"Punkty o niskiej pewności (<0.5): {(confidences < 0.5).sum():,} ({(confidences < 0.5).sum()/len(confidences)*100:.1f}%)")
    print(f"Punkty o wysokiej pewności (>0.8): {(confidences > 0.8).sum():,} ({(confidences > 0.8).sum()/len(confidences)*100:.1f}%)")
    print("="*80)

    # Expected results for bridge/viaduct
    print("\n💡 OCZEKIWANE WYNIKI DLA MOSTU:")
    print("   ✅ Zabudowania (most): 60-70%")
    print("   ✅ Droga (nawierzchnia): 10-20%")
    print("   ✅ Zieleń (drzewa): 5-10%")
    print("   ✅ Bariery: 3-5%")
    print("   ✅ Reszta: <5%")

    # Check if results look reasonable
    print("\n🔍 WERYFIKACJA:")

    zabudowania_pct = next((pct for name, _, pct in results if "Zabudowania" in name), 0)
    droga_pct = next((pct for name, _, pct in results if "Droga" in name), 0)
    znaki_pct = next((pct for name, _, pct in results if "Znaki" in name), 0)

    success = True

    if zabudowania_pct > 40:
        print(f"   ✅ Zabudowania: {zabudowania_pct:.1f}% (sensowne dla mostu)")
    else:
        print(f"   ⚠️ Zabudowania: {zabudowania_pct:.1f}% (za mało dla mostu)")
        success = False

    if droga_pct > 5:
        print(f"   ✅ Droga: {droga_pct:.1f}% (sensowne dla nawierzchni)")
    else:
        print(f"   ⚠️ Droga: {droga_pct:.1f}% (za mało)")

    if znaki_pct < 30:
        print(f"   ✅ Znaki Drogowe: {znaki_pct:.1f}% (OK - nie dominują)")
    else:
        print(f"   ❌ Znaki Drogowe: {znaki_pct:.1f}% (ZA DUŻO! Model nie działa poprawnie!)")
        success = False

    if confidences.mean() > 0.5:
        print(f"   ✅ Średnia pewność: {confidences.mean():.2%} (OK)")
    else:
        print(f"   ⚠️ Średnia pewność: {confidences.mean():.2%} (niska)")

    print("\n" + "="*80)
    if success:
        print("🎉 MODEL DZIAŁA POPRAWNIE!")
        print("   Most klasyfikowany głównie jako Zabudowania ✅")
        print("   Wyniki są sensowne dla infrastruktury mostowej")
    else:
        print("❌ MODEL NIE DZIAŁA POPRAWNIE!")
        print("   Prawdopodobnie:")
        print("   1. Weights nie załadowały się poprawnie")
        print("   2. Używa random weights zamiast pre-trained")
        print("   3. Problem z class mapping")
        print("\n   Debug:")
        print(f"   - using_sensaturban: {inference_engine.use_sensaturban}")
        print(f"   - device: {inference_engine.device}")
    print("="*80)

    return success


if __name__ == "__main__":
    try:
        success = test_sensaturban_model()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
