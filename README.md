# TensorFlow Practice Project


# Notes

For issues with TensorFlow certificate try to edit:
/venv/lib/python3.8/site-packages/tensorflow/python/keras/utils/data_util.py

Add below imports:

```python
import requests
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
```