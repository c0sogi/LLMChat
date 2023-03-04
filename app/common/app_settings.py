from fastapi import FastAPI, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from typing import Union
from app.common.config import (
    LocalConfig,
    ProdConfig,
    TestConfig,
)
from app.database.connection import db
from app.middlewares.token_validator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routers import index, auth, services, users
from app.dependencies import api_service_dependency, user_dependency


def create_app(config: Union[LocalConfig, ProdConfig, TestConfig]) -> FastAPI:
    # App & DB
    new_app = FastAPI()
    db.init_app(app=new_app, config=config)

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
    new_app.include_router(index.router)
    new_app.include_router(auth.router, prefix="/api", tags=["auth"])
    new_app.include_router(
        services.router,
        prefix="/api",
        tags=["Services"],
        dependencies=[Depends(user_dependency)],
    ) if config.debug else new_app.include_router(
        services.router,
        prefix="/api",
        tags=["Services"],
        dependencies=[Depends(user_dependency), Depends(api_service_dependency)],
    )
    new_app.include_router(
        users.router,
        prefix="/api",
        tags=["Users"],
        dependencies=[Depends(user_dependency)],
    )
    return new_app


# # Test function for manual Let's encrypt validation challenge
# def create_app(config: Union[LocalConfig, ProdConfig, TestConfig]) -> FastAPI:
#     # App & DB
#     new_app = FastAPI()
#     new_app.mount("/", StaticFiles(directory="/home/ubuntu/app/"))
#     return new_app
