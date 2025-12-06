#!/bin/bash
# Test runner script for CPK Point Cloud Classifier

# Activate virtual environment
source venv/bin/activate

# Run integration tests
echo "Running integration tests..."
python -c "import sys; sys.path.insert(0, '.'); exec(open('tests/test_integration.py').read())"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
