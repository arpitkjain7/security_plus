from sqlalchemy import Column, String, Integer, BigInteger

from db.base import Base


class Batch(Base):
    __tablename__ = "batch"

    batch_id = Column(String, primary_key=True)
    licence_num = Column(String)
    licence_plate_image = Column(String)
    created = Column(String)
    updated = Column(String)

    def __init__(self, batch_id, licence_num, licence_plate_image, created, updated):
        self.batch_id = batch_id
        self.licence_num = licence_num
        self.licence_plate_image = licence_plate_image
        self.created = created
        self.updated = updated
