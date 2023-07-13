from datetime import datetime, date, timedelta
from re import match


class UTC:
    def __init__(self):
        self.utc_now = datetime.utcnow()

    @classmethod
    def now(cls, hour_diff: int = 0) -> datetime:
        return cls().utc_now + timedelta(hours=hour_diff)

    @classmethod
    def now_isoformat(cls, hour_diff: int = 0) -> str:
        return (cls().utc_now + timedelta(hours=hour_diff)).isoformat() + "Z"

    @classmethod
    def date(cls, hour_diff: int = 0) -> date:
        return cls.now(hour_diff=hour_diff).date()

    @classmethod
    def timestamp(cls, hour_diff: int = 0) -> int:
        return int(cls.now(hour_diff=hour_diff).strftime("%Y%m%d%H%M%S"))

    @classmethod
    def timestamp_to_datetime(cls, timestamp: int, hour_diff: int = 0) -> datetime:
        return datetime.strptime(str(timestamp), "%Y%m%d%H%M%S") + timedelta(hours=hour_diff)

    @classmethod
    def date_code(cls, hour_diff: int = 0) -> int:
        return int(cls.date(hour_diff=hour_diff).strftime("%Y%m%d"))

    @staticmethod
    def check_string_valid(string: str) -> bool:
        """Check if a string is in ISO 8601 UTC datetime format.
        e.g. 2023-05-22T05:08:29.087279Z" -> True
            2023-05-22T05:08:29Z -> True
            2023-05-22T05:08:29 -> False
            2023/05/22T05:08:29.087279Z -> False"""
        regex = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
        return True if match(regex, string) else False


if __name__ == "__main__":
    # 예제 사용 예
    print(UTC.check_string_valid("2023-05-22T05:08:29.087279Z"))  # True
    print(UTC.check_string_valid("2023-05-22T05:08:29Z"))  # True
    print(UTC.check_string_valid("2023-05-22T05:08:29"))  # False
    print(UTC.check_string_valid("2023/05/22T05:08:29.087279Z"))  # False
    print(UTC.timestamp_to_datetime(10000101000000))  # Prefix timestamp
    print(UTC.timestamp_to_datetime(99991231235959))  # Suffix timestamp
