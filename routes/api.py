from fastapi import FastAPI
from model_server import logger
from routes import (
    infer_router,
    train_router,
)

from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import os

logging = logger(__name__)
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
work_dir = os.environ.get("work_dir", "/app/temp")
work_dir = os.path.join(work_dir, "configuration-data")
os.makedirs(work_dir, exist_ok=True)
api.mount(work_dir, StaticFiles(directory=work_dir), name="configuration-data")


@api.get("/ping")
def ping():
    return {"document": "ping from model server"}


api.include_router(infer_router.infer_router, tags=["infer"])
api.include_router(train_router.train_router, tags=["train"])
