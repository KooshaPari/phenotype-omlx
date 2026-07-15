"""Build the JNI bridge between TurboQuant+ Rust and a JVM runtime."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

if sys.platform != "darwin" and sys.platform != "linux":
    sys.exit(f"python-bindings-jni only supports macOS + Linux, got {sys.platform}")

ext = Extension(
    "phenotype_jni",
    sources=["src/phenotype_jni.pyx", "src/jni_glue.c"],
    include_dirs=["src"],
    libraries=["jvm"],
    library_dirs=["${JAVA_HOME}/lib"],
    extra_compile_args=["-O3", "-std=c11"],
)

setup(
    name="phenotype-jni",
    version="0.1.0",
    ext_modules=cythonize([ext], language_level=3),
    requires=["Cython>=0.29", "setuptools"],
)
