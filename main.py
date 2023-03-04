import uvicorn
from app.common.app_settings import create_app
from app.common.config import Config


app = create_app(config=Config.get())
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
    # --ssl-keyfile=key.pem --ssl-certfile=cert.pem
