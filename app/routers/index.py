from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse("app/web/index.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/contents/favicon.ico")


@router.get("/test")
async def test(request: Request):
    return {"username": request.session.get("username", None)}


# @router.websocket("/test")
# async def send_generation(websocket: WebSocket):
#     try:
#         await websocket.accept()

#         # Create an asyncio.Queue and a simple queue
#         manager = mp.Manager()
#         m_queue, m_done = manager.Queue(), manager.Event()

#         await asyncio.gather(
#             asyncio.get_event_loop().run_in_executor(
#                 process_pool_executor,
#                 llama_cpp_generation,
#                 LLMModels.wizard_vicuna_13b.value,
#                 "hello!",
#                 m_queue,
#                 m_done,
#             ),
#             sending_task(websocket, m_queue, m_done),
#         )

#     except Exception as exception:
#         ApiLogger.cerror(exception, exc_info=True)
