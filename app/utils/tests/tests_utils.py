from uuid import uuid4


def random_user_generator():
    random_8_digits = str(hash(uuid4()))[:8]
    return {
        "email": f"{random_8_digits}@test.com",
        "password": "123",
        "name": f"{random_8_digits}",
        "phone_number": f"010{random_8_digits}",
    }
