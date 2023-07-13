from enum import Enum
from typing import Any, Type, TypeVar, Union


EnumMixinGeneric = TypeVar("EnumMixinGeneric", bound="EnumMixin")


class EnumMixin(Enum):
    @classmethod
    def __init_subclass__(cls) -> None:
        cls.__dynamic_member_map__: dict[str, Enum] = {}
        super().__init_subclass__()

    @classmethod
    @property
    def static_member_map(
        cls: Type[EnumMixinGeneric],
    ) -> dict[str, EnumMixinGeneric]:
        return cls._member_map_  # type: ignore

    @classmethod
    @property
    def dynamic_member_map(
        cls: Type[EnumMixinGeneric],
    ) -> dict[str, Enum]:
        return cls.__dynamic_member_map__

    @classmethod
    @property
    def member_map(
        cls: Type[EnumMixinGeneric],
    ) -> dict[str, EnumMixinGeneric | Enum]:
        return cls.static_member_map | cls.dynamic_member_map

    @classmethod
    @property
    def member_names(
        cls: Type[EnumMixinGeneric],
    ) -> list[str]:
        return list(cls.member_map.keys())

    @classmethod
    def add_member(
        cls: Type[EnumMixinGeneric],
        name: str,
        value: Any,
    ) -> None:
        # Create a temporary dict for old members and the new member
        temp_dict = {member.name: member.value for member in cls}
        temp_dict[name] = value

        # Dynamically create and replace old enum class with updated one.
        globals()[cls.__name__] = Enum(cls.__name__, temp_dict)

        # Update dynamic members map
        cls.__dynamic_member_map__[name] = globals()[cls.__name__][name]

    @classmethod
    def get_name(
        cls: Type[EnumMixinGeneric],
        attribute: Union[EnumMixinGeneric, Enum, str],
    ) -> str:
        try:
            if isinstance(attribute, str):  # when attribute is string
                if attribute in cls.member_names:
                    return cls.member_map[attribute].name
                elif attribute.upper() in cls.member_map:
                    return cls.member_map[attribute.upper()].name
                return cls.member_map[attribute.lower()].name
            if (
                attribute in cls.member_map.values()
            ):  # when attribute is member
                return attribute.name
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"attribute must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    @classmethod
    def get_value(
        cls: Type[EnumMixinGeneric],
        attribute: Union[EnumMixinGeneric, Enum, str],
    ) -> Any:
        try:
            if isinstance(attribute, str):  # when attribute is string
                if attribute in cls.member_names:
                    return cls.member_map[attribute].value
                elif attribute.upper() in cls.member_map:
                    return cls.member_map[attribute.upper()].value
                return cls.member_map[attribute.lower()].value
            elif (
                attribute in cls.member_map.values()
            ):  # when attribute is member
                return attribute.value
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"attribute must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    @classmethod
    def get_member(
        cls: Type[EnumMixinGeneric],
        attribute: Union[EnumMixinGeneric, Enum, str],
    ) -> EnumMixinGeneric | Enum:
        try:
            if isinstance(attribute, str):  # when attribute is string
                if attribute in cls.member_names:
                    return cls.member_map[attribute]
                elif attribute.upper() in cls.member_map:
                    return cls.member_map[attribute.upper()]
                return cls.member_map[attribute.lower()]
            elif (
                attribute in cls.member_map.values()
            ):  # when attribute is member
                return attribute
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"attribute must be a string or an instance of {cls.__name__}, got {attribute}"
            )

    @classmethod
    def get_static_member(
        cls: Type[EnumMixinGeneric], attribute: Union[EnumMixinGeneric, str]
    ) -> EnumMixinGeneric:
        try:
            if isinstance(attribute, str):  # when attribute is string
                if attribute in cls.member_names:
                    return cls.static_member_map[attribute]
                elif attribute.upper() in cls.static_member_map:
                    return cls.static_member_map[attribute.upper()]
                return cls[attribute.lower()]
            elif isinstance(attribute, cls):  # when attribute is member
                return attribute
            else:
                raise TypeError
        except (KeyError, TypeError):
            raise ValueError(
                f"attribute must be a string or an instance of {cls.__name__}, got {attribute}"
            )
