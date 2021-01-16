from fastapi import FastAPI
from routes import (
    infer_router,
    train_router,
)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


app.include_router(infer_router.infer_router, tags=["infer"])
app.include_router(train_router.train_router, tags=["train"])
