from starlette_admin.fields import (
    EnumField,
    BooleanField,
    EmailField,
    PhoneField,
    StringField,
    NumberField,
    DateTimeField,
    PasswordField,
)
from starlette_admin.contrib.sqla.view import ModelView

from app.database.schemas.auth import Users

USER_STATUS_TYPES = [
    ("admin", "Admin"),
    ("active", "Active"),
    ("deleted", "Deleted"),
    ("blocked", "Blocked"),
]
API_KEY_STATUS_TYPES = [
    ("active", "Active"),
    ("deleted", "Deleted"),
    ("stopped", "Stopped"),
]





class UserAdminView(ModelView):
    fields = [
        EnumField("status", choices=USER_STATUS_TYPES, select2=False),
        NumberField("id"),
        EmailField("email"),
        PasswordField("password"),
        StringField("name"),
        PhoneField("phone_number"),
        BooleanField("marketing_agree"),
        DateTimeField("created_at"),
        DateTimeField("updated_at"),
        StringField("ip_address"),
        "api_keys",
    ]
    fields_default_sort = [Users.id, ("id", False)]


class ApiKeyAdminView(ModelView):
    fields = [
        EnumField("status", choices=API_KEY_STATUS_TYPES, select2=False),
        NumberField("id"),
        StringField("access_key"),
        StringField("user_memo"),
        BooleanField("is_whitelisted"),
        DateTimeField("created_at"),
        DateTimeField("updated_at"),
        "users",
        "whitelists",
    ]
    fields_default_sort = [Users.id, ("id", False)]
