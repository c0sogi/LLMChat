import subprocess
import sys
from logging import Logger, getLogger
from pathlib import Path
from typing import Optional

LIB_BASE_NAME: str = "llama"
REPOSITORY_FOLDER: str = "repositories"
PROJECT_GIT_URL: str = "https://github.com/abetlen/llama-cpp-python.git"
PROJECT_NAME: str = "llama_cpp"
MODULE_NAME: str = "llama_cpp"
VENDOR_GIT_URL: str = "https://github.com/ggerganov/llama.cpp.git"
VENDOR_NAME: str = "llama.cpp"
CMAKE_CONFIG: str = "Release"
SCRIPT_FILE_NAME: str = "build-llama-cpp"
CMAKE_OPTIONS: dict[str, str] = {
    "cublas": "-DBUILD_SHARED_LIBS=ON -DLLAMA_CUBLAS=ON",
    "default": "-DBUILD_SHARED_LIBS=ON",
}


WINDOWS_BUILD_SCRIPT = r"""
cd {vendor_path}
rmdir /s /q build
mkdir build
cd build
cmake .. {cmake_args}
cmake --build . --config {cmake_config}
cd ../../../../..
"""

UNIX_BUILD_SCRIPT = r"""#!/bin/bash
cd {vendor_path}
rm -rf build
mkdir build
cd build
cmake .. {cmake_args}
cmake --build . --config {cmake_config}
cd ../../../../..
"""
REPOSITORY_PATH: Path = Path(REPOSITORY_FOLDER).resolve()
PROJECT_PATH: Path = REPOSITORY_PATH / Path(PROJECT_NAME)
MODULE_PATH: Path = PROJECT_PATH / Path(MODULE_NAME)
VENDOR_PATH: Path = PROJECT_PATH / Path("vendor") / Path(VENDOR_NAME)
BUILD_OUTPUT_PATH: Path = (
    VENDOR_PATH / Path("build") / Path("bin") / Path(CMAKE_CONFIG)
)


def _clone_repositories() -> None:
    if not PROJECT_PATH.exists():
        REPOSITORY_PATH.mkdir(exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--recurse-submodules",
                PROJECT_GIT_URL,
                PROJECT_NAME,
            ],
            cwd=REPOSITORY_PATH,
        )

    if not VENDOR_PATH.exists():
        PROJECT_PATH.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", VENDOR_GIT_URL],
            cwd=Path(PROJECT_PATH),
        )


def _get_lib_paths() -> list[Path]:
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        return [
            MODULE_PATH / f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        return [
            MODULE_PATH / f"lib{LIB_BASE_NAME}.so",
            MODULE_PATH / f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        return [
            MODULE_PATH / f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")


def _get_build_lib_paths() -> list[Path]:
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        return [
            BUILD_OUTPUT_PATH / f"lib{LIB_BASE_NAME}.so",
        ]
    elif sys.platform == "darwin":
        return [
            BUILD_OUTPUT_PATH / f"lib{LIB_BASE_NAME}.so",
            BUILD_OUTPUT_PATH / f"lib{LIB_BASE_NAME}.dylib",
        ]
    elif sys.platform == "win32":
        return [
            BUILD_OUTPUT_PATH / f"{LIB_BASE_NAME}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")


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
        return "cp"
    elif sys.platform == "darwin":
        return "cp"
    elif sys.platform == "win32":
        return "copy"
    else:
        raise RuntimeError("Unsupported platform")


def _get_script_content(cmake_args: str) -> str:
    if sys.platform.startswith("linux"):
        return UNIX_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH,
            cmake_args=cmake_args,
            cmake_config=CMAKE_CONFIG,
        )
    elif sys.platform == "darwin":
        return UNIX_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH,
            cmake_args=cmake_args,
            cmake_config=CMAKE_CONFIG,
        )
    elif sys.platform == "win32":
        return WINDOWS_BUILD_SCRIPT.format(
            vendor_path=VENDOR_PATH,
            cmake_args=cmake_args,
            cmake_config=CMAKE_CONFIG,
        )
    else:
        raise RuntimeError("Unsupported platform")


def build_shared_lib(logger: Optional[Logger] = None) -> None:
    """
    Ensure that the llama.cpp DLL exists.
    You need cmake and Visual Studio 2019 to build llama.cpp.
    You can download cmake here: https://cmake.org/download/
    """

    if logger is None:
        logger = getLogger(__name__)
        logger.setLevel("INFO")

    if not PROJECT_PATH.exists():
        _clone_repositories()

    if not any(lib_path.exists() for lib_path in _get_lib_paths()):
        logger.critical("🦙 llama.cpp DLL not found, building it...")
        script_extension = _get_script_extension()
        copy_command = _get_copy_command()
        build_lib_paths = _get_build_lib_paths()

        script_paths: list[Path] = []
        for script_name, cmake_args in CMAKE_OPTIONS.items():
            if sys.platform == "darwin" and "cublas" in cmake_args.lower():
                logger.warning(
                    "🦙 cuBLAS is not supported on macOS, skipping this..."
                )
                continue
            MODULE_PATH.mkdir(exist_ok=True)
            script_path = MODULE_PATH / Path(
                f"build-llama-cpp-{script_name}.{script_extension}"
            )
            script_paths.append(script_path)
            script_content = _get_script_content(cmake_args)
            for build_lib_path in build_lib_paths:
                script_content += (
                    f"\n{copy_command} {build_lib_path} {MODULE_PATH}"
                )

            with open(script_path, "w") as f:
                f.write(script_content)

        is_built: bool = False
        for script_path in script_paths:
            try:
                # Try to build with cublas.
                logger.critical(
                    f"🦙 Trying to build llama.cpp DLL: {script_path}"
                )
                subprocess.run([script_path], check=True)
                logger.critical("🦙 llama.cpp DLL successfully built!")
                is_built = True
                break
            except subprocess.CalledProcessError:
                logger.critical("🦙 Could not build llama.cpp DLL!")
        if not is_built:
            raise RuntimeError("🦙 Could not build llama.cpp DLL!")


if __name__ == "__main__":
    build_shared_lib()
