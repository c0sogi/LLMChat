from app.mixins.enum import EnumMixin


class Lotties(EnumMixin):
    CLICK = "lottie-click"
    READ = "lottie-read"
    SCROLL_DOWN = "lottie-scroll-down"
    GO_BACK = "lottie-go-back"
    SEARCH_WEB = "lottie-search-web"
    SEARCH_DOC = "lottie-search-doc"
    OK = "lottie-ok"
    FAIL = "lottie-fail"
    TRANSLATE = "lottie-translate"

    def format(self, contents: str, end: bool = True) -> str:
        return f"\n```{self.get_value(self)}\n{contents}" + (
            "\n```\n" if end else ""
        )


if __name__ == "__main__":
    print(Lotties.CLICK.format("hello"))
