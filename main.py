from dotenv import load_dotenv

load_dotenv("./.env")
import uvicorn
from app.common.app_settings import create_app
from app.common.config import config

app = create_app(config=config)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
