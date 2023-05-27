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
from app.common.config import JWT_SECRET, Config
from app.database.connection import cache, db
from app.database.schemas.auth import ApiKeys, ApiWhiteLists, Users
from app.dependencies import USER_DEPENDENCY, api_service_dependency
from app.middlewares.token_validator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routers import auth, index, services, users, websocket
from app.shared import Shared
from app.utils.chat.cache_manager import CacheManager
from app.utils.js_initializer import js_url_initializer
from app.utils.logger import api_logger
from app.viewmodels.admin import ApiKeyAdminView, UserAdminView


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

    # Admin
    if db.engine is not None:
        admin = Admin(
            db.engine,
            title="Admin Console",
            auth_provider=MyAuthProvider(),
            middlewares=[Middleware(SessionMiddleware, secret_key=JWT_SECRET)],
        )
        admin.add_view(UserAdminView(Users, icon="fa fa-users", label="Users"))
        admin.add_view(ApiKeyAdminView(ApiKeys, icon="fa fa-key", label="API Keys"))
        admin.add_view(ModelView(ApiWhiteLists, icon="fa fa-list", label="API White Lists"))
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
        dependencies=[Depends(USER_DEPENDENCY)],
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
            import asyncio

            import uvloop  # type: ignore

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            api_logger.critical("uvloop installed!")
        except ImportError:
            api_logger.critical("uvloop not installed!")

    @new_app.on_event("shutdown")
    async def shutdown():
        # await CacheManager.delete_user(f"testaccount@{HOST_MAIN}")
        Shared().process_manager.shutdown()
        Shared().process_pool_executor.shutdown()
        await db.close()
        await cache.close()
        api_logger.critical("DB & CACHE connection closed!")

    return new_app
