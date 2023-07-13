from threading import Thread

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette_admin.contrib.sqla.admin import Admin
from starlette_admin.contrib.sqla.view import ModelView
from starlette_admin.views import DropDown, Link

from app.auth.admin import MyAuthProvider
from app.database.connection import cache, db
from app.database.schemas.auth import ApiKeys, ApiWhiteLists, Users
from app.dependencies import USER_DEPENDENCY, api_service_dependency
from app.middlewares.token_validator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routers import auth, index, services, user_services, users, websocket
from app.shared import Shared
from app.utils.chat.managers.cache import CacheManager
from app.utils.js_initializer import js_url_initializer
from app.utils.logger import ApiLogger
from app.viewmodels.admin import ApiKeyAdminView, UserAdminView

from .app_settings_llama_cpp import monitor_llama_cpp_server
from .config import JWT_SECRET, Config


async def on_startup():
    """
    Performs necessary operations during application startup.

    This function is called when the application is starting up.
    It performs the following operations:
    - Logs a startup message using ApiLogger.
    - Retrieves the configuration object.
    - Checks if the MySQL database connection is initiated and logs the status.
    - Raises a ConnectionError if the Redis cache connection is not established.
    - Checks if the Redis cache connection is initiated and logs the status.
    - Attempts to import and set uvloop as the event loop policy, if available, and logs the result.
    - Starts Llama CPP server monitoring if the Llama CPP completion URL is provided.
    """
    ApiLogger.ccritical("⚙️ Booting up...")
    config = Config.get()
    shared = Shared()
    if db.is_initiated:
        ApiLogger.ccritical("MySQL DB connected!")
    else:
        ApiLogger.ccritical("MySQL DB connection failed!")
    if cache.redis is None:
        raise ConnectionError("Redis is not connected yet!")
    if cache.is_initiated and await cache.redis.ping():
        await CacheManager.delete_user(f"testaccount@{config.host_main}")
        ApiLogger.ccritical("Redis CACHE connected!")
    else:
        ApiLogger.ccritical("Redis CACHE connection failed!")
    try:
        import asyncio

        import uvloop  # type: ignore

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        ApiLogger.ccritical("uvloop installed!")
    except ImportError:
        ApiLogger.ccritical("uvloop not installed!")

    if config.llama_completion_url:
        # Start Llama CPP server monitoring
        ApiLogger.ccritical("Llama CPP server monitoring started!")
        shared.thread = Thread(
            target=monitor_llama_cpp_server,
            args=(config, shared),
            daemon=True,
        )
        shared.thread.start()


async def on_shutdown():
    """
    Performs necessary operations during application shutdown.

    This function is called when the application is shutting down.
    It performs the following operations:
    - Logs a shutdown message using ApiLogger.
    - Sets terminate signals for shared threads and processes.
    - Shuts down the process manager, if available.
    - Shuts down the process pool executor, if available.
    - Terminates and joins the process, if available.
    - Joins the thread, if available.
    - Closes the database and cache connections.
    - Logs a message indicating the closure of DB and CACHE connections.
    """
    ApiLogger.ccritical("⚙️ Shutting down...")
    shared = Shared()
    # await CacheManager.delete_user(f"testaccount@{HOST_MAIN}")
    shared.thread_terminate_signal.set()
    shared.process_terminate_signal.set()

    process_manager = shared._process_manager
    if process_manager is not None:
        process_manager.shutdown()

    process_pool_executor = shared._process_pool_executor
    if process_pool_executor is not None:
        process_pool_executor.shutdown(wait=True)

    process = shared._process
    if process is not None:
        process.terminate()
        process.join()

    thread = shared._thread
    if thread is not None:
        thread.join()

    await db.close()
    await cache.close()
    ApiLogger.ccritical("DB & CACHE connection closed!")


def create_app(config: Config) -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Args:
        config (Config): The configuration object.

    Returns:
        FastAPI: The configured FastAPI application.

    This function creates a new FastAPI application, sets the specified title, description, and version,
    and adds `on_startup` and `on_shutdown` event handlers.

    It then starts the database and cache connections and initializes the JavaScript URL.

    If the database engine is available, it adds admin views for managing users, API keys, and API white lists.

    Next, it adds the necessary middlewares for access control, CORS, and trusted hosts.

    It mounts the "/chat" endpoint for serving static files, and includes routers for index, websocket,
    authentication, services, users, and user services.

    Finally, it sets the application's config and shared state and returns the configured application.
    """
    # Initialize app & db & js
    new_app = FastAPI(
        title=config.app_title,
        description=config.app_description,
        version=config.app_version,
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
    )
    db.start(config=config)
    cache.start(config=config)
    js_url_initializer(js_location="app/web/main.dart.js")

    # Admin
    if db.engine is not None:
        admin = Admin(
            db.engine,
            title="Admin Console",
            auth_provider=MyAuthProvider(),
            middlewares=[Middleware(SessionMiddleware, secret_key=JWT_SECRET)],
        )
        admin.add_view(UserAdminView(Users, icon="fa fa-users", label="Users"))
        admin.add_view(
            ApiKeyAdminView(ApiKeys, icon="fa fa-key", label="API Keys")
        )
        admin.add_view(
            ModelView(
                ApiWhiteLists, icon="fa fa-list", label="API White Lists"
            )
        )
        admin.add_view(
            DropDown(
                "Links",
                icon="fa fa-link",
                views=[
                    Link("Index", url="/"),
                    Link("Docs", url="/docs"),
                    Link("Chat", url="/chat", target="_blank"),
                ],
            )
        )
        admin.mount_to(new_app)

    # Middlewares
    """
    Access control middleware: Authorized request only
    CORS middleware: Allowed sites only
    Trusted host middleware: Allowed host only
    """

    new_app.add_middleware(
        dispatch=access_control, middleware_class=BaseHTTPMiddleware
    )
    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_sites,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    new_app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.trusted_hosts,
        except_path=["/health"],
    )

    # Routers
    new_app.mount("/chat", StaticFiles(directory="./app/web", html=True))
    new_app.include_router(index.router, tags=["index"])
    new_app.include_router(websocket.router, prefix="/ws", tags=["websocket"])
    new_app.include_router(
        auth.router,
        prefix="/api",
        tags=["auth"],
    )
    new_app.include_router(
        services.router,
        prefix="/api",
        tags=["Services"],
        dependencies=[Depends(api_service_dependency)],
    )
    new_app.include_router(
        users.router,
        prefix="/api",
        tags=["Users"],
        dependencies=[Depends(USER_DEPENDENCY)],
    )
    new_app.include_router(
        user_services.router,
        prefix="/api",
        tags=["User Services"],
        dependencies=[Depends(USER_DEPENDENCY)],
    )
    new_app.state.config = config
    new_app.state.shared = Shared()

    @new_app.get("/health")
    async def health():
        return "ok"

    return new_app
