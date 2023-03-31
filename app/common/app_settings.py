import logging
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from app.common.config import (
    LocalConfig,
    ProdConfig,
    TestConfig,
)
from app.database.schema import db
from app.middlewares.token_validator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routers import index, auth, services, users, ws
from app.dependencies import (
    api_service_dependency,
    user_dependency,
)
from app.utils.js_initializer import js_url_initializer


def create_app(config: LocalConfig | ProdConfig | TestConfig) -> FastAPI:
    js_url_initializer(js_location="app/web/main.dart.js")
    # App & DB
    new_app = FastAPI(
        title=config.app_title,
        description=config.app_description,
        version=config.app_version,
    )
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
    new_app.mount("/chatgpt", StaticFiles(directory="./app/web", html=True))
    new_app.include_router(index.router, tags=["index"])
    new_app.include_router(ws.router, prefix="/ws", tags=["websocket"])
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
        # self.engine.connect()
        logging.critical(">>> DB connected")

    @new_app.on_event("shutdown")
    async def shutdown():
        await db.session.remove()
        await db.engine.dispose()
        logging.critical(">>> DB disconnected")

    return new_app
