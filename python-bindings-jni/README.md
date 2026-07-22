# python-bindings-jni

Cython bridge between TurboQuant+ and a **JVM runtime** (Java/Kotlin/Scala).
Useful when the host language for inference is JVM rather than C/C++.

## Build

```bash
# Prereqs: Python 3.10+, Cython 0.29+, JDK 17+
pip install setuptools Cython

# Point to the JVM library location
export JAVA_HOME=$(/usr/libexec/java_home)            # macOS
export PHENOTYPE_JNI_LIBS="$PWD/runtime"

# Compile the Cython module (calls gcc/clang to embed libjvm)
python setup.py build_ext --inplace
```

## Use from Python

```python
from phenotype_jni import JNIEncoder
enc = JNIEncoder(jvm_path="/path/to/libjvm.dylib")
packed, scales, zeros = enc.encode([0.1, -0.2, 0.3, ...], bits=4, group_size=64)
recon = enc.decode(packed, scales, zeros, n=64, bits=4, group_size=64)
```

## Fallback

If `JNIEncoder(...)` fails to initialize (no JVM installed, or `jvm_path` wrong),
Python callers automatically fall back to the pure-Python `refract-quant` package
already shipped with `turboquant_plus`.
