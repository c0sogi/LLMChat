import requests
from typing import Type, Callable
import os  # 윈도우 절전모드 활성화
from pynput.keyboard import Listener
from concurrent import futures
from threading import Event


class EventListener:
    def __init__(
        self,
        listener_type: Type[Listener],
        callbacks: list[dict[str, Callable[..., None]]],
        timeout: int | None,
    ):
        self.num_threads: int = callbacks.__len__()
        self.listener_type: Listener = listener_type
        self.callbacks: list[dict[str, Callable[..., None]]] = callbacks
        self.timeout: int | None = timeout
        self.events: list[Event] = [Event() for _ in range(self.num_threads)]

    def callback_wrapper(self, func) -> None:
        def wrapper(callback_idx: int):
            func()
            self.events[callback_idx].set()
            print("SET!!!")

        return wrapper

    def event_handler(self, listener: Listener, callback_idx: int) -> bool | Exception:
        try:
            listener.start()
            result = self.events[callback_idx].wait(timeout=self.timeout)
        except Exception as e:
            return e
        else:
            return result
        finally:
            listener.stop()

    def run(self) -> bool:
        wrapped_callbacks = ...
        with futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            ensured_futures = [
                executor.submit(
                    self.event_handler,
                    listener=self.listener_type(**self.callback_wrapper[callback_idx]),
                    callback_idx=callback_idx,
                )
                for callback_idx in range(self.num_threads)
            ]
            future_results = [
                future.result() for future in futures.as_completed(ensured_futures)
            ]
        print(future_results)
        return True


def on_press(key):
    print("pressed", key)
    ...


def sleep_if_idle(sleep_timer: int) -> None:
    listener = EventListener(
        listener_type=Listener, callbacks=[{"on_press": on_press}], timeout=sleep_timer
    )
    is_key_pressed: bool | Exception = listener.run()

    if is_key_pressed is False:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")  # 절전모드 활성화 명령어 실행
    elif isinstance(is_key_pressed, Exception):
        ...
    else:
        ...


def call_scale_api(message):
    data = {"input": {"input": message}}
    headers = {"Authorization": "Basic clffnxo9r00igtf1ayzapnbww"}
    response = requests.post(
        "https://dashboard.scale.com/spellbook/api/v2/deploy/nz43bfa",
        json=data,
        headers=headers,
    )
    return response.json()


def start_chat():
    print("챗봇과 대화를 시작합니다. 종료하려면 'exit'를 입력하세요.d")

    while True:
        user_input = input("나: ")
        if user_input.lower() == "exit":
            print("채팅을 종료합니다.")
            break

        response = call_scale_api(user_input)
        print("챗봇: ", response["output"])


if __name__ == "__main__":
    sleep_if_idle(5)
