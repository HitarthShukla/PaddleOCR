import subprocess
import sys
from pathlib import Path


ENV = Path("/content/paddle-env")


def run(cmd):
    print("\n" + "=" * 70)
    print("RUNNING:", " ".join(map(str, cmd)))
    print("=" * 70)
    subprocess.check_call(cmd)


print("=" * 70)
print("TABLE-OCR COLAB SETUP")
print("=" * 70)

# ------------------------------------------------------------
# 1. Create Python 3.12 environment if it doesn't exist
# ------------------------------------------------------------

if not ENV.exists():
    print("Creating Python 3.12 virtual environment...")

    run([
        "python3.12",
        "-m",
        "venv",
        str(ENV)
    ])
else:
    print("Existing Python environment found.")


PYTHON = str(ENV / "bin" / "python")


# ------------------------------------------------------------
# 2. Upgrade pip tooling
# ------------------------------------------------------------

run([
    PYTHON,
    "-m",
    "pip",
    "install",
    "-q",
    "-U",
    "pip",
    "wheel",
])


# ------------------------------------------------------------
# 3. Paddle GPU
# ------------------------------------------------------------

run([
    PYTHON,
    "-m",
    "pip",
    "install",
    "-q",
    "paddlepaddle-gpu",
    "-i",
    "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
])


# ------------------------------------------------------------
# 4. PaddleOCR
# ------------------------------------------------------------

run([
    PYTHON,
    "-m",
    "pip",
    "install",
    "-q",
    "paddlex[ocr]",
])

run([
    PYTHON,
    "-m",
    "pip",
    "install",
    "-q",
    "paddleocr",
])

# ------------------------------------------------------------
# 5. Other project dependencies
# ------------------------------------------------------------

run([
    PYTHON,
    "-m",
    "pip",
    "install",
    "-q",
    "pandas",
    "numpy",
    "opencv-python-headless",
])


# ------------------------------------------------------------
# 6. Verify Paddle
# ------------------------------------------------------------

run([
    PYTHON,
    "-c",
    """
import paddle

print("Paddle:", paddle.__version__)
print("CUDA compiled:", paddle.is_compiled_with_cuda())
print("Device:", paddle.device.get_device())

if not paddle.is_compiled_with_cuda():
    raise RuntimeError(
        "Paddle is NOT using CUDA. "
        "GPU installation failed."
    )

if not paddle.device.get_device().startswith("gpu"):
    raise RuntimeError(
        "Paddle is not running on the GPU."
    )

print("GPU CHECK PASSED")
"""
])


# ------------------------------------------------------------
# 7. Verify PaddleOCR
# ------------------------------------------------------------

run([
    PYTHON,
    "-c",
    """
import paddleocr
print("PaddleOCR:", paddleocr.__version__)
"""
])


print()
print("=" * 70)
print("SETUP COMPLETE")
print("=" * 70)
print()
print("Run your OCR with:")
print()
print("    /content/paddle-env/bin/python test_table_ocr.py")
print()