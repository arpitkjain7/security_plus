from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from infer.inference import infer
import os
from datetime import datetime

infer_router = APIRouter()


@infer_router.post("/get_licence_plate")
async def infer_image(file: UploadFile = File(...)):
    batch_id = str(int(datetime.now().timestamp() * 1000))
    folder_path = os.path.join("work_dir", batch_id)
    file_path = os.path.join(folder_path, file.filename)
    os.makedirs(folder_path, exist_ok=True)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    text = infer(image_path=file_path, batch_id=batch_id)
    return {"Licence Number": text}
