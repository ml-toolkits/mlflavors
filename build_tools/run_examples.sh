#!/bin/bash
echo "Running Orbit example"
python examples/orbit/train.py

echo "Running Sktime example"
python examples/sktime/train.py

echo "Running StatsForecast example"
python examples/statsforecast/train.py

echo "Running PyOD example"
python examples/pyod/train.py

echo "Running SDV example"
python examples/sdv/train.py
