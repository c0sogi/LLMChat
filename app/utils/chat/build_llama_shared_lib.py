from logging import Logger, getLogger
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


LIB_BASE_NAME: str = "llama"
REPOSITORY_LOCATION: str = "./repositories/llama_cpp"

BASE_PATH: Path = Path(f"{REPOSITORY_LOCATION}/llama_cpp/").resolve()
VENDOR_PATH: Path = Path(f"{REPOSITORY_LOCATION}/vendor/llama.cpp").resolve()
BUILD_PATH: Path = Path(
    f"{REPOSITORY_LOCATION}/vendor/llama.cpp/build/bin/release"
).resolve()

CMAKE_OPTIONS: dict[str, str] = {
    "cublas": "-DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON",  # First build cublas, then build the rest
    "default": "-DBUILD_SHARED_LIBS=ON",
}
SCRIPT_FILE_NAME: str = "build-llama-cpp"
WINDOWS_BUILD_SCRIPT = r"""
cd {vendor_path}
rmdir /s /q build
mkdir build
cd build
cmake .. {cmake_args}
cmake --build . --config Release
cd ../../../../..
"""

UNIX_BUILD_SCRIPT = r"""#!/bin/bash
cd {vendor_path}
rm -rf build
mkdir build
cd build
cmake .. {cmake_args}
cmake --build . --config Release
cd ../../../../..
"""


def _get_lib_paths() -> list[Path]:
    _lib_paths: list[Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            BASE_PATH / f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            BASE_PATH / f"lib{LIB_BASE_NAME}.so",
            BASE_PATH / f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            BASE_PATH / f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")
    return _lib_paths


def _get_build_lib_paths() -> list[Path]:
    _build_lib_paths: list[Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _build_lib_paths += [
            BUILD_PATH / f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        _build_lib_paths += [
            BUILD_PATH / f"lib{LIB_BASE_NAME}.so",
            BUILD_PATH / f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        _build_lib_paths += [
            BUILD_PATH / f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")
    return _build_lib_paths


def _get_script_extension() -> str:
    if sys.platform.startswith("linux"):
        return "sh"
    elif sys.platform == "darwin":
        return "sh"
    elif sys.platform == "win32":
        return "bat"
    else:
        raise RuntimeError("Unsupported platform")


def _get_copy_command() -> str:
    if sys.platform.startswith("linux"):
        copy_command = "cp"
    elif sys.platform == "darwin":
        copy_command = "cp"
    elif sys.platform == "win32":
        copy_command = "copy"
    else:
        raise RuntimeError("Unsupported platform")
    return copy_command


def _get_script_content(cmake_args: str) -> str:
    if sys.platform.startswith("linux"):
        script_content = UNIX_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH, cmake_args=cmake_args
        )
    elif sys.platform == "darwin":
        script_content = UNIX_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH, cmake_args=cmake_args
        )
    elif sys.platform == "win32":
        script_content = WINDOWS_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH, cmake_args=cmake_args
        )
    else:
        raise RuntimeError("Unsupported platform")
    return script_content


def build_shared_lib(logger: Optional[Logger] = None) -> None:
    """
    Ensure that the llama.cpp DLL exists.
    You need cmake and Visual Studio 2019 to build llama.cpp.
    You can download cmake here: https://cmake.org/download/
    """

    if logger is None:
        logger = getLogger(__name__)
        logger.setLevel("INFO")

    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(
            "🦙 Could not find llama-cpp-python repositories folder!"
        )

    if not any(lib_path.exists() for lib_path in _get_lib_paths()):
        logger.critical("🦙 llama.cpp DLL not found, building it...")
        script_extension = _get_script_extension()
        copy_command = _get_copy_command()
        build_lib_paths = _get_build_lib_paths()

        script_paths: list[Path] = []
        for script_name, cmake_args in CMAKE_OPTIONS.items():
            if sys.platform == "darwin" and "cublas" in cmake_args.lower():
                logger.warning(
                    "🦙 cuBLAS is not supported on macOS, skipping this build option..."
                )
                continue
            script_path = BASE_PATH / Path(
                f"build-llama-cpp-{script_name}.{script_extension}"
            )
            script_paths.append(script_path)
            script_content = _get_script_content(cmake_args)
            for build_lib_path in build_lib_paths:
                script_content += f"\n{copy_command} {build_lib_path} {BASE_PATH}"

            with open(script_path, "w") as f:
                f.write(script_content)

        is_built: bool = False
        for script_path in script_paths:
            try:
                # Try to build with cublas. cuBLAS is a CUDA library that speeds up matrix multiplication.
                logger.critical(f"🦙 Trying to build llama.cpp DLL: {script_path}")
                subprocess.run([script_path]).check_returncode()
                logger.critical("🦙 llama.cpp DLL successfully built!")
                is_built = True
                break
            except subprocess.CalledProcessError:
                logger.critical("🦙 Could not build llama.cpp DLL!")
        if not is_built:
            raise RuntimeError("🦙 Could not build llama.cpp DLL!")
