# ChatGPT Web App Server 🎉

Welcome to the ChatGPT Web App Server repository, a full-stack implementation of an API server built with Python FastAPI, and a beautiful frontend powered by Flutter. This project is designed to deliver a seamless chat experience with the advanced GPT-3 model, offering a modern infrastructure that can be easily extended when GPT-4's Multimodal and Plugin features become available. Enjoy your stay! 🚀

## Key Features

- FastAPI - High-performance web framework for building APIs with Python.
- Flutter - Cross-platform UI toolkit for creating stunning native apps with a single codebase.
- Middlewares - Token validation and authentication to keep your API secure.
- Database Connection - Manage database connections and execute SQL queries with SQLAlchemy and Redis.
- Database CRUD Operations - Easily perform Create, Read, Update, and Delete actions.
- ChatGPT WebSocket Connection - Real-time, two-way communication with the ChatGPT model.
- Behind the ChatGPT WebSocket Connection - Well-structured API backend with cache and message management.
- ChatGPT Text Generation - Seamless integration with the OpenAI API for text generation and message management.

## Getting Started / Installation

To set up the ChatGPT Web App Server on your local machine, follow these simple steps:

1. Clone the repository:

```bash
git clone https://github.com/c0sogi/chatgpt-webapp-server.git
```

2. Change to the project directory:

```bash
cd chatgpt-webapp-server
```

3. Create `.env` file and setup for fastapi server, referring to `.env-sample` file. Enter Database connection to create, OpenAI API Key, and other necessary configurations. Optionals are not required, just leave them blank.

4. To run the server, execute. It may take a few minutes to start the server for the first time:

```bash
docker-compose -f docker-compose-local.yaml up -d
```

5. To stop the server, execute:

```bash
docker-compose -f docker-compose-local.yaml down
```

6. Now you can access the server at `http://localhost:8000/docs` and the database at `db:3306` or `cache:6379`. You can also access the ChatGPT Web App at `http://localhost:8000/chatgpt`.


Your ChatGPT Web App Server should now be up and running, ready to provide an engaging chat experience!

## License

This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution, as long as the original copyright and license notice are included in any copy or substantial portion of the software.


# Quick Summary of FastAPI

`FastAPI` is a modern web framework for building APIs with Python.
It has high performance, easy to learn, fast to code, and ready for production.
One of the main features of `FastAPI` is that it supports concurrency and `async`/`await` syntax.
This means that you can write code that can handle multiple tasks at the same time without blocking each other,
especially when dealing with I/O bound operations, such as network requests, database queries, file operations, etc.

`FastAPI` supports both asynchronous and synchronous endpoints.
If you define an endpoint with an `async def`, every time you access it, a new coroutine is created in the main event loop,
allowing you to manage tasks concurrently. For example:
```python
@app.get("/async")
async def async_endpoint():
    # You can use await inside an `async def` function to call other async functions
    # or libraries that support async operations
    results = await some_async_library()
    return results
```
However, if you need to use a third-party library that doesn't support asynchronous operations within the endpoint,you may block other users' tasks. To address this, you can consider the following solutions:

### (1) Define the endpoint with a `def` rather than an `async def`.
- This creates a new thread pool each time a user connects to the endpoint, and ensures that different tasks work on different threads to maintain concurrency. For example:

```python
@app.get("/sync")
def sync_endpoint():
    # You can use normal functions or libraries that don't support async operations
    results = some_sync_library()
    return results
```

### (2) Inside the async endpoint, create a new threadpool using the `fastapi.concurrency.run_in_threadpool` function. 
- The principle is the same. For example:

```python
@app.get("/mixed")
async def mixed_endpoint():
    # You can use await with fastapi.concurrency.run_in_threadpool
    # to run sync functions or libraries in a separate thread pool
    results = await run_in_threadpool(some_sync_library)
    return results
```
However, these two methods are not always good because you can't create an infinite number of threadpools, and CPU have to manage the threads each time you create a threadpool, which can cause unnecessary overhead. The best approach is to use asynchronous libraries to achieve concurrency, and use the await statement within asynchronous endpoints.

`FastAPI` is based on `AnyIO`, which makes it compatible with both Python's standard library asyncio and Trio.In particular, you can directly use `AnyIO` for your advanced concurrency use cases that require more advanced patterns in your own code.



# Quick summary of Flutter

`Flutter` is an open-source UI toolkit developed by Google for building native user interfaces for mobile, web, and desktop platforms from a single codebase. It uses `Dart`, a modern object-oriented programming language, and provides a rich set of customizable widgets that can adapt to any design.


## Features and Benefits of Flutter

Here are some of the main features and benefits of using Flutter for your next app project:

- **Cross-platform development**: Flutter allows you to create apps that run on `Android`, `iOS`, `web`, and `desktop` platforms using the same codebase. This saves you time and money by avoiding the need to maintain multiple codebases and hire different developers for each platform.
- **Same UI and business logic in all platforms**: Flutter uses its own rendering engine, called `Skia`, to draw widgets directly on the screen. This means that your app will look and behave the same way on any device, regardless of the operating system or screen size. You can also customize your widgets to fit your brand identity and design guidelines.
- **Fast development cycle**: Flutter supports `hot reload` and hot restart features, which allow you to see the changes in your code instantly on the emulator or device without losing the app state or restarting the app. This enables you to iterate faster and fix bugs quicker.
- **Highly customizable UI design**: Flutter provides a wide variety of rich widgets that are built-in, animated, and responsive. You can also create your own custom widgets or use third-party packages from pub.dev, the official package repository for Flutter and Dart. You can also use Flutter's `declarative` style of programming to build complex UIs with less code and more readability.
- **Access to native features**: Flutter allows you to access native features and services of each platform, such as camera, geolocation, sensors, etc., by using platform channels. Platform channels are a mechanism for communicating between Dart code and native code using asynchronous messages. You can also use existing native libraries or write your own native code in `Java`, `Kotlin`, `Objective-C`, or `Swift`.
- **High performance**: Flutter apps run at 60 frames per second (FPS) or higher, which ensures smooth animations and transitions. Flutter also uses `ahead-of-time (AOT)` compilation to compile Dart code into native machine code, which improves the startup time and eliminates the performance issues caused by interpreters or virtual machines.
- **Strong community support**: Flutter has a large and active community of developers who contribute to its development and improvement. You can find many resources online, such as documentation, tutorials, blogs, videos, podcasts, etc., to help you learn and use Flutter. You can also join forums, groups, meetups, or events to connect with other Flutter enthusiasts and experts.

# Middlewares

This project uses `token_validator` middleware and other middlewares used in the FastAPI application. These middlewares are responsible for controlling access to the API, ensuring only authorized and authenticated requests are processed.

## Examples

The following middlewares are added to the FastAPI application:

1. Access Control Middleware: Ensures that only authorized requests are processed.
2. CORS Middleware: Allows requests from specific origins, as defined in the app configuration.
3. Trusted Host Middleware: Ensures that requests are coming from trusted hosts, as defined in the app configuration.

### Access Control Middleware

The Access Control Middleware is defined in the `token_validator.py` file. It is responsible for validating API keys and JWT tokens.

#### State Manager

The `StateManager` class is used to initialize request state variables. It sets the request time, start time, IP address, and user token.

#### Access Control

The `AccessControl` class contains two static methods for validating API keys and JWT tokens:

1. `api_service`: Validates API keys by checking the existence of required query parameters and headers in the request. It calls the `Validator.api_key` method to verify the API key, secret, and timestamp.
2. `non_api_service`: Validates JWT tokens by checking the existence of the 'authorization' header or 'Authorization' cookie in the request. It calls the `Validator.jwt` method to decode and verify the JWT token.

#### Validator

The `Validator` class contains two static methods for validating API keys and JWT tokens:

1. `api_key`: Verifies the API access key, hashed secret, and timestamp. Returns a `UserToken` object if the validation is successful.
2. `jwt`: Decodes and verifies the JWT token. Returns a `UserToken` object if the validation is successful.

#### Access Control Function

The `access_control` function is an asynchronous function that handles the request and response flow for the middleware. It initializes the request state using the `StateManager` class, determines the type of authentication required for the requested URL (API key or JWT token), and validates the authentication using the `AccessControl` class. If an error occurs during the validation process, an appropriate HTTP exception is raised.

### Token

Token utilities are defined in the `token.py` file. It contains two functions:

1. `create_access_token`: Creates a JWT token with the given data and expiration time.
2. `token_decode`: Decodes and verifies a JWT token. Raises an exception if the token is expired or cannot be decoded.

### Params Utilities

The `params_utils.py` file contains a utility function for hashing query parameters and secret key using HMAC and SHA256:

1. `hash_params`: Takes query parameters and secret key as input and returns a base64 encoded hashed string.

### Date Utilities

The `date_utils.py` file contains the `UTC` class with utility functions for working with dates and timestamps:

1. `now`: Returns the current UTC datetime with an optional hour difference.
2. `timestamp`: Returns the current UTC timestamp with an optional hour difference.
3. `timestamp_to_datetime`: Converts a timestamp to a datetime object with an optional hour difference.

### Logger

The `logger.py` file contains the `api_logger` function, which logs API request and response information, including the request URL, method, status code, client information, processing time, and error details (if applicable). The logger function is called at the end of the `access_control` function to log the processed request and response.

## Usage

To use the `token_validator` middleware in your FastAPI application, simply import the `access_control` function and add it as a middleware to your FastAPI instance:

```python
from app.middlewares.token_validator import access_control

app = FastAPI()

app.add_middleware(dispatch=access_control, middleware_class=BaseHTTPMiddleware)
```

Make sure to also add the CORS and Trusted Host middlewares for complete access control:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_sites,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.trusted_hosts,
    except_path=["/health"],
)
```

Now, any incoming requests to your FastAPI application will be processed by the `token_validator` middleware and other middlewares, ensuring that only authorized and authenticated requests are processed.

# Database Connection

This module `app.database.connection` provides an easy-to-use interface for managing database connections and executing SQL queries using SQLAlchemy and Redis. It supports MySQL, and can be easily integrated with this project.

## Features

- Create and drop databases
- Create and manage users
- Grant privileges to users
- Execute raw SQL queries
- Manage database sessions with async support
- Redis caching support for faster data access


## Usage

First, import the required classes from the module:

```python
from app.database.connection import MySQL, SQLAlchemy, RedisFactory
```

Next, create an instance of the `SQLAlchemy` class and configure it with your database settings:

```python
from app.common.config import Config

config: LocalConfig | TestConfig | ProdConfig = Config.get()
db = SQLAlchemy()
db.start(config)
```

Now you can use the `db` instance to execute SQL queries and manage sessions:

```python
# Execute a raw SQL query
result = await db.execute("SELECT * FROM users")

# Use the run_in_session decorator to manage sessions
@db.run_in_session
async def create_user(session, username, password):
    await session.execute("INSERT INTO users (username, password) VALUES (:username, :password)", {"username": username, "password": password})

await create_user("JohnDoe", "password123")
```

To use Redis caching, create an instance of the `RedisFactory` class and configure it with your Redis settings:

```python
cache = RedisFactory()
cache.start(config)
```

You can now use the `cache` instance to interact with Redis:

```python
# Set a key in Redis
await cache.redis.set("my_key", "my_value")

# Get a key from Redis
value = await cache.redis.get("my_key")
```
In fact, in this project, the `MySQL` class does the initial setup at app startup, and all database connections are made with only the `db` and `cache` variables present at the end of the module. 😅 

All db settings will be done in `create_app()` in `app.common.app_settings`.
For example, the `create_app()` function in `app.common.app_settings` will look like this:

```python
def create_app(config: LocalConfig | ProdConfig | TestConfig) -> FastAPI:
    # Initialize app & db & js
    new_app = FastAPI(
        title=config.app_title,
        description=config.app_description,
        version=config.app_version,
    )
    db.start(config=config)
    cache.start(config=config)
    js_url_initializer(js_location="app/web/main.dart.js")
    # Register routers
    # ...
    return new_app
```

# Database CRUD Operations

This project uses simple and efficient way to handle database CRUD (Create, Read, Update, Delete) operations using SQLAlchemy and two module and path: `app.database.models.schema` and `app.database.crud`. 

## Overview

### app.database.models.schema

The `schema.py` module is responsible for defining database models and their relationships using SQLAlchemy. It includes a set of classes that inherit from `Base`, an instance of `declarative_base()`. Each class represents a table in the database, and its attributes represent columns in the table. These classes also inherit from a `Mixin` class, which provides some common methods and attributes for all the models.

#### Mixin Class

The Mixin class provides some common attributes and methods for all the classes that inherit from it. Some of the attributes include:

- `id`: Integer primary key for the table.
- `created_at`: Datetime for when the record was created.
- `updated_at`: Datetime for when the record was last updated.
- `ip_address`: IP address of the client that created or updated the record.

It also provides several class methods that perform CRUD operations using SQLAlchemy, such as:

- `add_all()`: Adds multiple records to the database.
- `add_one()`: Adds a single record to the database.
- `update_where()`: Updates records in the database based on a filter.
- `fetchall_filtered_by()`: Fetches all records from the database that match the provided filter.
- `one_filtered_by()`: Fetches a single record from the database that matches the provided filter.
- `first_filtered_by()`: Fetches the first record from the database that matches the provided filter.
- `one_or_none_filtered_by()`: Fetches a single record or returns `None` if no records match the provided filter.

### app.database.crud
The `users.py` and `api_keys.py` module contains a set of functions that perform CRUD operations using the classes defined in `schema.py`. These functions use the class methods provided by the Mixin class to interact with the database.

Some of the functions in this module include:

- `create_api_key()`: Creates a new API key for a user.
- `get_api_keys()`: Retrieves all API keys for a user.
- `get_api_key_owner()`: Retrieves the owner of an API key.
- `get_api_key_and_owner()`: Retrieves an API key and its owner.
- `update_api_key()`: Updates an API key.
- `delete_api_key()`: Deletes an API key.
- `is_email_exist()`: Checks if an email exists in the database.
- `get_me()`: Retrieves user information based on user ID.
- `is_valid_api_key()`: Checks if an API key is valid.
- `register_new_user()`: Registers a new user in the database.
- `find_matched_user()`: Finds a user with a matching email in the database.

## Usage

To use the provided CRUD operations, import the relevant functions from the `crud.py` module and call them with the required parameters. For example:

```python
import asyncio
from app.database.crud.users import register_new_user, get_me, is_email_exist
from app.database.crud.api_keys import create_api_key, get_api_keys, update_api_key, delete_api_key

async def main():
    # Register a new user
    new_user = await register_new_user(email="test@test.com", hashed_password="...")

    # Get user information
    user = await get_me(user_id=1)

    # Check if an email exists in the database
    email_exists = await is_email_exist(email="test@test.com")

    # Create a new API key for user with ID 1
    new_api_key = await create_api_key(user_id=1, additional_key_info={"user_memo": "Test API Key"})

    # Get all API keys for user with ID 1
    api_keys = await get_api_keys(user_id=1)

    # Update the first API key in the list
    updated_api_key = await update_api_key(updated_key_info={"user_memo": "Updated Test API Key"}, access_key_id=api_keys[0].id, user_id=1)

    # Delete the first API key in the list
    await delete_api_key(access_key_id=api_keys[0].id, access_key=api_keys[0].access_key, user_id=1)

if __name__ == "__main__":
    asyncio.run(main())
```

# ChatGPT WebSocket Connection

You can access `ChatGPT` through `WebSocket` connection using two modules: `app/routers/websocket` and `app/utils/chatgpt/chatgpt_stream_manager`. These modules facilitate the communication between the `Flutter` client and the ChatGPT model through a WebSocket. With the WebSocket, you can establish a real-time, two-way communication channel to interact with the ChatGPT model.

## websocket.py

`websocket.py` is responsible for setting up a WebSocket connection and handling user authentication. It defines the WebSocket route `/chatgpt/{api_key}` that accepts a WebSocket and an API key as parameters.

When a client connects to the WebSocket, it first checks the API key to authenticate the user. If the API key is valid, the `begin_chat()` function is called from the `chatgpt_stream_manager.py` module to start the ChatGPT conversation.

In case of an unregistered API key or an unexpected error, an appropriate message is sent to the client and the connection is closed.

```python
@router.websocket("/chatgpt/{api_key}")
async def ws_chatgpt(websocket: WebSocket, api_key: str):
    ...
```

## chatgpt_stream_manager.py

`chatgpt_stream_manager.py` is responsible for managing the ChatGPT conversation and handling user messages. It defines the `begin_chat()` function, which takes a WebSocket, a user ID, and an OpenAI API key as parameters.

The function first initializes the user's GPT context from the cache manager. Then, it sends the initial message history to the client through the WebSocket.

The conversation continues in a loop until the connection is closed. During the conversation, the user's messages are processed and GPT's responses are generated accordingly.

```python
async def begin_chat(
    websocket: WebSocket,
    user_id: str,
    openai_api_key: str,
) -> None:
    ...
```

### Handling User Messages and Commands

User messages are processed using the `HandleMessage` class. If the message starts with `/`, it is treated as a command and the appropriate command response is generated. Otherwise, the user's message is processed and sent to the GPT model for generating a response.

```python
if msg.startswith("/"):
    ...
else:
    await HandleMessage.user(...)
await HandleMessage.gpt(...)
```

### Sending Messages to WebSocket

The `SendToWebsocket` class is used for sending messages and streams to the WebSocket. It has two methods: `message()` and `stream()`. The `message()` method sends a complete message to the WebSocket, while the `stream()` method sends a stream to the WebSocket.

```python
class SendToWebsocket:
    @staticmethod
    async def message(...):
        ...

    @staticmethod
    async def stream(...):
        ...
```

### Handling GPT Responses

The `HandleMessage` class also handles GPT responses. The `gpt()` method sends the GPT response to the WebSocket. If translation is enabled, the response is translated using the Google Translate API before sending it to the client.

```python
class HandleMessage:
    ...
    @staticmethod
    async def gpt(...):
        ...
```

### Handling Custom Commands

Commands are handled using the `get_command_response()` function. It takes the user's message and GPT context as parameters and executes the corresponding callback function depending on the command. 

```python
async def get_command_response(msg: str, user_gpt_context: UserGptContext) -> str | None:
    ...
```
You can add new commands by simply adding callback in `ChatGptCommands` class from `app.utils.chatgpt.chatgpt_commands`. It can accessed by client by sending a message starting with `/`, such as `/YOUR_CALLBACK_NAME`.
## Usage

To start a ChatGPT conversation, connect to the WebSocket route `/ws/chatgpt/{api_key}` with a valid API key registered in the database. Note that this API key is not the same as OpenAI API key, but only available for your server to validate the user. Once connected, you can send messages and commands to interact with the ChatGPT model. The WebSocket will send back GPT's responses in real-time. This websocket connection is established via Flutter app, which can accessed with endpoint `/chatgpt`.


# Behind the ChatGPT WebSocket Connection...

This project aims to create an API backend to enable the ChatGPT chatbot service. It utilizes a cache manager to store messages and user profiles in Redis, and a message manager to safely cache messages so that the number of tokens does not exceed an acceptable limit.

## Cache Manager

The Cache Manager (`ChatGptCacheManager`) is responsible for handling user context information and message histories. It stores these data in Redis, allowing for easy retrieval and modification. The manager provides several methods to interact with the cache, such as:

- `read_context`: Reads the user's GPT context from Redis.
- `create_context`: Creates a new user GPT context in Redis.
- `reset_context`: Resets the user's GPT context to default values.
- `update_message_histories`: Updates the message histories for a specific role (user, GPT, or system).
- `lpop_message_history` / `rpop_message_history`: Removes and returns the message history from the left or right end of the list.
- `append_message_history`: Appends a message history to the end of the list.
- `get_message_history`: Retrieves the message history for a specific role.
- `delete_message_history`: Deletes the message history for a specific role.
- `set_message_history`: Sets a specific message history for a role and index.

## Message Manager

The Message Manager (`MessageManager`) ensures that the number of tokens in message histories does not exceed the specified limit. It safely handles adding, removing, and setting message histories in the user's GPT context while maintaining token limits. The manager provides several methods to interact with message histories, such as:

- `add_message_history_safely`: Adds a message history to the user's GPT context, ensuring that the token limit is not exceeded.
- `rpop_message_history_safely`: Removes and returns the message history from the right end of the list while updating the token count.
- `set_message_history_safely`: Sets a specific message history in the user's GPT context, updating the token count and ensuring that the token limit is not exceeded.

## Usage

To use the cache manager and message manager in your project, import them as follows:

```python
from app.utils.chatgpt.chatgpt_cache_manager import chatgpt_cache_manager
from app.utils.chatgpt.chatgpt_message_manager import MessageManager
```

Then, you can use their methods to interact with the Redis cache and manage message histories according to your requirements.

For example, to create a new user GPT context:

```python
user_id = "example_user_id"
default_context = UserGptContext.construct_default(user_id=user_id)
await chatgpt_cache_manager.create_context(user_id=user_id, user_gpt_context=default_context)
```

To safely add a message history to the user's GPT context:

```python
user_gpt_context = await chatgpt_cache_manager.read_context(user_id)
content = "This is a sample message."
role = "user"  # can be "user", "gpt", or "system"
await MessageManager.add_message_history_safely(user_gpt_context, content, role)
```

# ChatGPT Text Generation Module

This module provides the functionality needed to integrate the OpenAI API with the ChatGPT chatbot service. It handles the process of organizing message history, generating text from the OpenAI API, and managing the asynchronous streaming of generated text.

## Overview

The main components of the module are:

1. `message_history_organizer`: Organizes message history for the OpenAI API.
2. `generate_from_openai`: Generates text from the OpenAI API and streams the response.

### message_history_organizer

This function takes a `UserGptContext` object and a boolean `send_to_openai` as input. It organizes the message history for the OpenAI API by appending system, user, and GPT message histories. If the context is discontinued, it appends a continuation message to the last GPT message.

```python
def message_history_organizer(
    user_gpt_context: UserGptContext, send_to_openai: bool = True
) -> list[dict]:
```

### generate_from_openai

This asynchronous generator function takes an `openai_api_key` and a `UserGptContext` object as input. It initializes an HTTP client with a timeout and continuously generates text from the OpenAI API. It also handles various exceptions that might occur during the generation process.

```python
async def generate_from_openai(
    openai_api_key: str,  # api key for openai
    user_gpt_context: UserGptContext,  # gpt context for user
) -> AsyncGenerator:  # async generator for streaming
```

## Usage

The module can be used for integrating the OpenAI API with your ChatGPT chatbot service as follows:

1. Import the required components:

```python
from app.utils.chatgpt.chatgpt_generation import message_history_organizer, generate_from_openai
```

2. Organize the message history:

```python
message_histories = message_history_organizer(user_gpt_context)
```

3. Generate text from the OpenAI API:

```python
async for generated_text in generate_from_openai(openai_api_key, user_gpt_context):
    # Process the generated text
```

4. Handle exceptions that may occur during text generation:

```python
try:
    # Generate text from OpenAI API
except GptLengthException:
    # Handle token limit exceeded
except GptContentFilterException:
    # Handle content filter exception
except GptConnectionException:
    # Handle connection error
except httpx.TimeoutException:
    # Handle timeout exception
except Exception as exception:
    # Handle unexpected exceptions
```

With these components, you can effectively integrate the OpenAI API with your ChatGPT chatbot service and generate text from the API based on the user's message history.