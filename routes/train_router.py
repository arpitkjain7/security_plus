from fastapi import APIRouter, BackgroundTasks, UploadFile, File

train_router = APIRouter()


@train_router.post("/train_object_detection_model")
async def train_model():
    return "test"
