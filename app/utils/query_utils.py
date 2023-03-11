from typing import List


def row_to_dict(model, *args, exclude: List = None):
    q_dict = {}
    for c in model.__table__.columns:
        if not args or c.name in args:
            if not exclude or c.name not in exclude:
                q_dict[c.name] = getattr(model, c.name)

    return q_dict


def parse_params(params: dict) -> str:
    return "&".join([f"{key}={value}" for key, value in params.items()])
