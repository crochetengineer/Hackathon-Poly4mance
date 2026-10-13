# Reaction Mechanism Discovery Challenge -- repository contract.
# Judges run these four, in order, on a clean machine. Keep them working.

# install dependencies and the local package (editable)
install:
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -e .

# write submission/predictions.csv for the held-out conditions
predict:
    python baseline/predict.py

# write submission/campaign.yaml (the experiments you would run next)
suggest:
    python baseline/suggest.py

# regenerate every figure used in the presentation
replicate:
    python baseline/replicate.py
