from contextlib import asynccontextmanager
from multiprocessing import Process
from os import kill
from signal import SIGINT
from threading import Event
from urllib import parse

import requests
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.shared import Shared
from app.utils.logger import ApiLogger

from .config import Config


def check_health(url: str) -> bool:
    """Check if the given url is available or not"""
    try:
        schema = parse.urlparse(url).scheme
        netloc = parse.urlparse(url).netloc
        if requests.get(f"{schema}://{netloc}/health").status_code != 200:
            return False
        return True
    except Exception:
        return False


def start_llama_cpp_server(config: Config, shared: Shared):
    """Start Llama CPP server. if it is already running, terminate it first."""

    if shared.process.is_alive():
        ApiLogger.cwarning("Terminating existing Llama CPP server")
        shared.process.terminate()
        shared.process.join()

    if config.llama_server_port is None:
        raise NotImplementedError("Llama CPP server port is not set")

    ApiLogger.ccritical("Starting Llama CPP server")
    shared.process = Process(
        target=run_llama_cpp, args=(config.llama_server_port,), daemon=True
    )
    shared.process.start()


def shutdown_llama_cpp_server(shared: Shared):
    """Shutdown Llama CPP server."""
    ApiLogger.ccritical("Shutting down Llama CPP server")
    if shared.process.is_alive() and shared.process.pid:
        kill(shared.process.pid, SIGINT)
        shared.process.join()


def monitor_llama_cpp_server(
    config: Config,
    shared: Shared,
) -> None:
    """Monitors the Llama CPP server and handles server availability.

    Parameters:
    - `config: Config`: An object representing the server configuration.
    - `shared: Shared`: An object representing shared data."""
    thread_sigterm: Event = shared.thread_terminate_signal
    if not config.llama_completion_url:
        return
    while True:
        if not check_health(config.llama_completion_url):
            if thread_sigterm.is_set():
                break
            if config.is_llama_booting:
                continue
            ApiLogger.cerror("Llama CPP server is not available")
            config.is_llama_available = False
            config.is_llama_booting = True
            try:
                start_llama_cpp_server(config=config, shared=shared)
            except (ImportError, NotImplementedError):
                ApiLogger.cerror(
                    "ImportError: Llama CPP server is not available"
                )
                return
            except Exception:
                ApiLogger.cexception(
                    "Unknown error: Llama CPP server is not available"
                )
                config.is_llama_booting = False
                continue
        else:
            config.is_llama_booting = False
            config.is_llama_available = True
    shutdown_llama_cpp_server(shared)


@asynccontextmanager
async def lifespan_llama_cpp(app: FastAPI):
    ApiLogger.ccritical("🦙 Llama.cpp server is running")
    yield
    ApiLogger.ccritical("🦙 Shutting down llama.cpp server...")


def create_app_llama_cpp():
    from app.routers import v1

    new_app = FastAPI(
        title="🦙 llama.cpp Python API",
        version="0.0.1",
        lifespan=lifespan_llama_cpp,
    )
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @new_app.get("/health")
    async def health():
        return "ok"

    new_app.include_router(v1.router)
    return new_app


def run_llama_cpp(port: int) -> None:
    from uvicorn import Config, Server

    from maintools import initialize_before_launch

    initialize_before_launch()

    Server(
        config=Config(
            create_app_llama_cpp(),
            host="0.0.0.0",
            port=port,
            log_level="warning",
        )
    ).run()


if __name__ == "__main__":
    run_llama_cpp(port=8002)
