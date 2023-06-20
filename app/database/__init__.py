from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta
from typing import TypeVar

Base: DeclarativeMeta = declarative_base()
TableGeneric = TypeVar("TableGeneric")
