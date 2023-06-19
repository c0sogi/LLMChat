import enum


class UserStatus(str, enum.Enum):
    admin = "admin"
    active = "active"
    deleted = "deleted"
    blocked = "blocked"


class ApiKeyStatus(str, enum.Enum):
    active = "active"
    stopped = "stopped"
    deleted = "deleted"
