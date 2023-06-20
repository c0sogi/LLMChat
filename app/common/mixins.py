from enum import Enum
from typing import Any, Type, TypeVar, Union


EnumGeneric = TypeVar("EnumGeneric", bound="EnumMixin")


class EnumMixin(Enum):
    @classmethod
    def get_name(cls: Type[EnumGeneric], attribute: Union[EnumGeneric, str]) -> str:
        try:
            if isinstance(attribute, cls):  # when role is member
                return attribute.name
            elif isinstance(attribute, str):  # when role is string
                if attribute in cls._member_names_:
                    return cls._member_map_[attribute].name
                elif attribute.upper() in cls._member_map_:
                    return cls._member_map_[attribute.upper()].name
                return cls[attribute.lower()].name
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"Role must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    @classmethod
    def get_value(cls: Type[EnumGeneric], attribute: Union[EnumGeneric, str]) -> Any:
        try:
            if isinstance(attribute, cls):  # when role is member
                return attribute.value
            elif isinstance(attribute, str):  # when role is string
                if attribute in cls._member_names_:
                    return cls._member_map_[attribute].value
                elif attribute.upper() in cls._member_map_:
                    return cls._member_map_[attribute.upper()].value
                return cls[attribute.lower()].value
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"Role must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    @classmethod
    def get_member(
        cls: Type[EnumGeneric], attribute: Union[EnumGeneric, str]
    ) -> EnumGeneric:
        try:
            if isinstance(attribute, cls):  # when role is member
                return attribute
            elif isinstance(attribute, str):  # when role is string
                if attribute in cls._member_names_:
                    return cls._member_map_[attribute]  # type: ignore
                elif attribute.upper() in cls._member_map_:
                    return cls._member_map_[attribute.upper()]  # type: ignore
                return cls[attribute.lower()]
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"Role must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    # @classmethod
    # def get_name(cls, role: Union["ChatRoles", str]) -> str:
    #     if isinstance(role, cls):  # when role is member
    #         return role.name
    #     elif not isinstance(role, str):
    #         raise ValueError(f"Invalid role: {role}")
    #     elif role in cls._value2member_map_:  # when role is value
    #         return cls._value2member_map_[role].name
    #     elif role.upper() in cls._member_map_:  # when role is name
    #         return role
    #     else:
    #         raise ValueError(f"Invalid role: {role}")

    # @classmethod
    # def get_value(cls, role: Union["ChatRoles", str]) -> str:
    #     if isinstance(role, cls):  # when role is member
    #         return role.value
    #     elif not isinstance(role, str):
    #         raise ValueError(f"Invalid role: {role}")
    #     elif role in cls._value2member_map_:  # when role is value
    #         return role
    #     elif role.upper() in cls._member_map_:  # when role is name
    #         return cls._member_map_[role.upper()].value
    #     else:
    #         raise ValueError(f"Invalid role: {role}")

    # @classmethod
    # def get_member(cls, role: Union["ChatRoles", str]) -> "ChatRoles":
    #     if isinstance(role, cls):  # when role is member
    #         return role
    #     elif role in cls._value2member_map_:  # when role is value
    #         return cls._value2member_map_[role]  # type: ignore
    #     elif not isinstance(role, str):
    #         raise ValueError(f"Invalid role: {role}")
    #     elif role.upper() in cls._member_map_:  # when role is name
    #         return cls._member_map_[role.upper()]  # type: ignore
    #     else:
    #         raise ValueError(f"Invalid role: {role}")
