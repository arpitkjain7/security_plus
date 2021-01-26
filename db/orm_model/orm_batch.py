from sqlalchemy import Column, String, Integer, BigInteger, UniqueConstraint

from db.base import Base


class Batch(Base):
    __tablename__ = "batch"
    row_id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String, unique=True, nullable=False)
    frame_id = Column(String, unique=True, nullable=False)
    licence_num = Column(String)
    licence_plate_image = Column(String)
    created = Column(String)
    updated = Column(String)
    __table_args__ = tuple(UniqueConstraint("batch_id", "frame_id", name="batch_un"))

    def __init__(
        self, batch_id, frame_id, licence_num, licence_plate_image, created, updated
    ):
        self.batch_id = batch_id
        self.frame_id = frame_id
        self.licence_num = licence_num
        self.licence_plate_image = licence_plate_image
        self.created = created
        self.updated = updated
