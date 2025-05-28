This package provides experimental integration of perceval and pytorch and in particular QuantumLayer object allowing smooth integration in pytorch AI workflows. See nn_models/README.md for documentation, and nn_models/examples/iris_classifier.py.

The following should work:

```bash
virtualenv venv
source venv/bin/activate
pip install -e pcvl_pytorch
pip install -e nn_models
pip install seaborn scikit-learn
python nn_models/examples/iris_classifier.py
```

Since this module is still experimental, please share with us any issue you can find.

Note that this packaged is not to be distributed or pushed on public repository.
