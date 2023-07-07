from app.models.llms import LLMModels


def test_dynamic_llm_load():
    print("- member_map keys -")
    print(LLMModels.member_map.keys())
    print("- static member map keys -")
    print(LLMModels.static_member_map.keys())
    print("- dynamic member map keys -")
    print(LLMModels.dynamic_member_map.keys())

    LLMModels.add_member("foo", {"bar": "baz"})

    print("- member_map keys -")
    print(LLMModels.member_map.keys())
    print("- static member map keys -")
    print(LLMModels.static_member_map.keys())
    print("- dynamic member map keys -")
    print(LLMModels.dynamic_member_map.keys())

    print("- member_map class -")
    print(LLMModels.member_map)

    print("- is dynamic member instance of LLMModels ? -")
    print(isinstance(LLMModels.dynamic_member_map["foo"], LLMModels))
    print(f"- {LLMModels.dynamic_member_map['foo']} -")
