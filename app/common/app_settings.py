from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from app.common.config import Config
from app.database.connection import db, cache
from app.middlewares.token_validator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routers import index, auth, services, users, websocket
from app.dependencies import (
    api_service_dependency,
    user_dependency,
)
from app.utils.logger import api_logger
from app.utils.chat.cache_manager import CacheManager
from app.utils.js_initializer import js_url_initializer
from app.dependencies import process_pool_executor


def create_app(config: Config) -> FastAPI:
    # Initialize app & db & js
    new_app = FastAPI(
        title=config.app_title,
        description=config.app_description,
        version=config.app_version,
    )
    db.start(config=config)
    cache.start(config=config)
    js_url_initializer(js_location="app/web/main.dart.js")

    # Middlewares
    """
    Access control middleware: Authorized request only
    CORS middleware: Allowed sites only
    Trusted host middleware: Allowed host only
    """
    new_app.add_middleware(dispatch=access_control, middleware_class=BaseHTTPMiddleware)
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
        dependencies=[Depends(user_dependency)],
    )

    @new_app.on_event("startup")
    async def startup():
        if db.is_initiated:
            api_logger.critical("MySQL DB connected!")
        else:
            api_logger.critical("MySQL DB connection failed!")
        if cache.redis is None:
            raise ConnectionError("Redis is not connected yet!")
        if cache.is_initiated and await cache.redis.ping():
            await CacheManager.delete_user(f"testaccount@{config.host_main}")
            api_logger.critical("Redis CACHE connected!")
        else:
            api_logger.critical("Redis CACHE connection failed!")
        try:
            import uvloop  # type: ignore
            import asyncio

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            api_logger.critical("uvloop installed!")
        except ImportError:
            api_logger.critical("uvloop not installed!")

    @new_app.on_event("shutdown")
    async def shutdown():
        process_pool_executor.shutdown(cancel_futures=True, wait=True)
        # await CacheManager.delete_user(f"testaccount@{HOST_MAIN}")
        await db.close()
        await cache.close()
        api_logger.critical("DB & CACHE connection closed!")

    return new_app
