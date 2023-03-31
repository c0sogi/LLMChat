from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.hashes import HashAlgorithm, SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime, timedelta
from jwt import decode as jwt_decode
from jwt import encode as jwt_encode
from jwt.exceptions import ExpiredSignatureError, DecodeError
from string import ascii_letters, digits
from secrets import choice
from uuid import uuid4
from os import urandom
from os.path import exists
from hmac import HMAC, new
from re import findall
from base64 import urlsafe_b64encode, b64encode
import json
from app.errors.api_exceptions import Responses_401
from app.database.schema import ApiKeys
from app.models.base_models import AddApiKey
from app.common.config import JWT_ALGORITHM, JWT_SECRET


class SecretConfigSetup:
    algorithm: HashAlgorithm = SHA256
    length: int = 32
    iterations: int = 100000

    def __init__(self, password: str, json_file_name: str) -> None:
        self.password = password
        self.json_file_name = json_file_name

    @classmethod
    def get_kdf(cls, salt: bytes) -> PBKDF2HMAC:
        return PBKDF2HMAC(
            salt=salt,
            algorithm=cls.algorithm,
            length=cls.length,
            iterations=cls.iterations,
        )

    def encrypt(self) -> None:
        # Derive an encryption key from the password and salt using PBKDF2
        salt = urandom(16)
        fernet = Fernet(
            urlsafe_b64encode(
                SecretConfigSetup.get_kdf(salt=salt).derive(
                    key_material=self.password.encode()
                )
            )
        )
        with open(self.json_file_name, "r") as f:
            plain_data = json.load(f)

        # Encrypt the string using the encryption key
        encrypted_data = fernet.encrypt(json.dumps(plain_data).encode())

        # Save the encrypted data and salt to file
        with open(f"{self.json_file_name}.enc", "wb") as f:
            f.write(salt)
            f.write(encrypted_data)

    def decrypt(self) -> any:
        # Load the encrypted data and salt from file
        with open(f"{self.json_file_name}.enc", "rb") as f:
            salt = f.read(16)
            fernet = Fernet(
                urlsafe_b64encode(
                    SecretConfigSetup.get_kdf(salt=salt).derive(
                        key_material=self.password.encode()
                    )
                )
            )
            encrypted_data = f.read()
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data)

    def initialize(self) -> any:
        if not exists(f"{self.json_file_name}.enc"):
            self.encrypt()

        while True:
            try:
                secret_config = self.decrypt()
            except InvalidToken:
                self.password = input("Wrong password! Enter password again: ")
            else:
                return secret_config


def encode_from_utf8(text):
    # Check if text contains any non-ASCII characters
    matches = findall(r"[^\x00-\x7F]+", text)
    if len(matches) == 0:
        # Return text if it doesn't contain any non-ASCII characters
        return text
    else:
        # Encode and replace non-ASCII characters with UTF-8 formatted text
        for match in matches:
            encoded_text = b64encode(match.encode("utf-8")).decode("utf-8")
            text = text.replace(match, "=?UTF-8?B?" + encoded_text + "?=")
        return text


def hash_params(query_params: str, secret_key: str) -> str:
    mac: HMAC = new(
        key=bytes(secret_key, encoding="utf-8"),
        msg=bytes(query_params, encoding="utf-8"),
        digestmod="sha256",
    )
    return str(b64encode(mac.digest()).decode("utf-8"))


async def generate_api_key(user_id: int, additional_key_info: AddApiKey) -> ApiKeys:
    alnums = ascii_letters + digits
    secret_key = "".join(choice(alnums) for _ in range(40))
    uid = f"{str(uuid4())[:-12]}{str(uuid4())}"
    new_api_key = ApiKeys(
        secret_key=secret_key,
        user_id=user_id,
        access_key=uid,
        **additional_key_info.dict(),
    )
    return new_api_key


def create_access_token(*, data: dict = None, expires_delta: int = None) -> str:
    to_encode: dict = data.copy()
    if expires_delta is not None and expires_delta != 0:
        to_encode.update({"exp": datetime.utcnow() + timedelta(hours=expires_delta)})
    return jwt_encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def token_decode(authorization: str) -> dict:
    try:
        authorization = authorization.replace("Bearer ", "")
        payload = jwt_decode(authorization, key=JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except ExpiredSignatureError:
        raise Responses_401.token_expired
    except DecodeError:
        raise Responses_401.token_decode_failure
    return payload


# from app.utils.encoding_and_hashing import SecretConfigSetup
# password_from_environ = environ.get("SECRET_CONFIGS_PASSWORD", None)
# secret_config_setup = SecretConfigSetup(
#     password=password_from_environ
#     if password_from_environ is not None
#     else input("Enter Passwords:"),
#     json_file_name="secret_configs.json",
# )
#
#
# @dataclass(frozen=True)
# class SecretConfig(metaclass=SingletonMetaClass):
#     secret_config: dict = field(default_factory=secret_config_setup.initialize)
