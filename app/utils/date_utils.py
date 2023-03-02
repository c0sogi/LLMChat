from datetime import datetime, date, timedelta


class UTC:
    def __init__(self):
        self.utc_now = datetime.utcnow()

    @classmethod
    def now(cls, hour_diff: int = 0) -> datetime:
        return cls().utc_now + timedelta(hours=hour_diff)

    @classmethod
    def date(cls, hour_diff: int = 0) -> date:
        return cls.now(hour_diff=hour_diff).date()

    @classmethod
    def timestamp(cls, hour_diff: int = 0) -> int:
        return int(cls.now(hour_diff=hour_diff).strftime("%y%m%d%H%M%S"))

    @classmethod
    def date_code(cls, hour_diff: int = 0) -> int:
        return int(cls.date(hour_diff=hour_diff).strftime("%Y%m%d"))
