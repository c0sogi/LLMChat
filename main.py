from os import environ
from maintools import initialize_before_launch


if __name__ == "__mp_main__":
    """Option 1: Skip section for multiprocess spawning
    This section will be skipped when running in multiprocessing mode"""
    pass

elif __name__ == "__main__":
    #
    """Option 2: Debug mode
    Running this file directly to debug the app
    Run this section if you don't want to run app in docker"""

    if environ.get("API_ENV") != "test":
        environ["API_ENV"] = "local"
    environ["DOCKER_MODE"] = "False"

    import uvicorn

    from app.common.app_settings import create_app
    from app.common.config import config

    initialize_before_launch()
    app = create_app(config=config)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
    )
else:
    """Option 3: Non-debug mode
    Docker will run this section as the main entrypoint
    This section will mostly be used.
    Maybe LLaMa won't work in Docker."""
    from app.common.app_settings import create_app
    from app.common.config import config

    initialize_before_launch()
    app = create_app(config=config)
