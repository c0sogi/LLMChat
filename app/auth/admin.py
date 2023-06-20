from starlette.requests import Request
from starlette.responses import Response
from starlette_admin.auth import AdminUser, AuthProvider
from starlette_admin.exceptions import FormValidationError, LoginFailed
from app.common.config import config


class MyAuthProvider(AuthProvider):
    async def login(
        self,
        username: str,
        password: str,
        remember_me: bool,
        request: Request,
        response: Response,
    ) -> Response:
        if len(username) < 3:
            """Form data validation"""
            raise FormValidationError({"username": "Ensure username has at least 03 characters"})

        if username == config.mysql_user and password == config.mysql_password:
            """Save `username` in session"""
            request.session.update({"username": username})
            return response

        raise LoginFailed("Invalid username or password")

    async def is_authenticated(self, request) -> bool:
        if request.session.get("username", None) == config.mysql_user:
            """
            Save current `user` object in the request state. Can be used later
            to restrict access to connected user.
            """
            return True

        return False

    def get_admin_user(self, request: Request) -> AdminUser:
        return AdminUser(username=config.mysql_user)

    async def logout(self, request: Request, response: Response) -> Response:
        request.session.clear()
        return response
