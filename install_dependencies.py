import subprocess
import sys
import os
import urllib.request
import tempfile

def install(package, no_deps=False):
    cmd = [sys.executable, "-m", "pip", "install", package]
    if no_deps:
        cmd.append("--no-deps")
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        raise e

def install_fairseq_wheel():
    print("\n--- Checking and Installing Precompiled fairseq Wheel ---")
    try:
        import fairseq
        print("fairseq is already installed.")
        return
    except ImportError:
        pass

    py_version = sys.version_info
    py_major = py_version.major
    py_minor = py_version.minor

    # Support cp311, cp312, cp313
    if (py_major, py_minor) not in [(3, 11), (3, 12), (3, 13)]:
        print(f"Python {py_major}.{py_minor} is not explicitly supported by precompiled wheels.")
        print("Attempting default pip install fairseq...")
        install("fairseq")
        return

    wheel_url = f"https://github.com/BlueAmulet/fairseq-win-whl/releases/download/ci_build/fairseq-0.13.2-cp{py_major}{py_minor}-cp{py_major}{py_minor}-win_amd64.whl"
    print(f"Downloading precompiled wheel for Python {py_major}.{py_minor}...")
    print(f"URL: {wheel_url}")

    temp_dir = tempfile.gettempdir()
    wheel_path = os.path.join(temp_dir, f"fairseq-0.13.2-cp{py_major}{py_minor}-cp{py_major}{py_minor}-win_amd64.whl")

    try:
        urllib.request.urlretrieve(wheel_url, wheel_path)
        print(f"Download complete: {wheel_path}")
        print("Installing wheel...")
        install(wheel_path)
        print("fairseq installed successfully.")
    except Exception as e:
        print(f"Failed to install precompiled wheel: {e}")
        print("Attempting default pip install fairseq...")
        try:
            install("fairseq")
        except Exception:
            print("Standard fairseq installation failed. Make sure Visual Studio C++ Build Tools are installed.")
    finally:
        if os.path.exists(wheel_path):
            try:
                os.remove(wheel_path)
            except Exception:
                pass

if __name__ == "__main__":
    print("====================================================")
    print("        ANONYVOX DEPENDENCY AUTO-INSTALLER          ")
    print("====================================================\n")
    try:
        # Install standard requirements
        install("sounddevice>=0.4.6")
        install("numpy>=1.24.0")
        
        # Install precompiled fairseq wheel
        install_fairseq_wheel()
        
        # Install rvc-python dependencies that have prebuilt wheels for CP311-CP313
        print("\n--- Installing rvc-python dependencies ---")
        deps = [
            "faiss-cpu", 
            "ffmpeg-python", 
            "loguru", 
            "av", 
            "soundfile", 
            "praat-parselmouth", 
            "torchcrepe", 
            "pyworld",
            "einops"
        ]
        for dep in deps:
            install(dep)
            
        # Install rvc-python itself with --no-deps to bypass the incompatible old numpy/faiss version pins
        print("\n--- Installing rvc-python (Bypassing conflicts) ---")
        install("rvc-python", no_deps=True)
        
        print("\n====================================================")
        print(" SUCCESS: All dependencies installed successfully!  ")
        print(" Run the application with: python app.py            ")
        print("====================================================")
    except Exception as e:
        print("\n====================================================")
        print(f" ERROR: Dependency installation failed: {e}        ")
        print("====================================================")
        sys.exit(1)
