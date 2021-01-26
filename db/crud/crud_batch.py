from db.base import Session
from db.orm_model.orm_batch import Batch
from datetime import datetime

session = Session()


def create_batch(request):
    # batch = Batch()
    print(f"request from infer: {request}")
    current_time = datetime.now()
    # batch.batch_id == request.get("batch_id")
    # batch.licence_num == request.get("licence_num")
    # batch.licence_plate_image == request.get("licence_plate_path")
    # batch.created == current_time
    # batch.updated == current_time
    batch = Batch(
        request.get("batch_id"),
        request.get("frame_id"),
        request.get("licence_num"),
        request.get("licence_plate_path"),
        current_time,
        current_time,
    )
    session.add(batch)
    session.commit()
    session.close()
