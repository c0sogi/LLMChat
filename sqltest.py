from typing import Any
from uuid import uuid4

from sqlalchemy import select
from app.database.schema import db, Users


if __name__ == "__main__":
    from asyncio import run

    async def main() -> None:
        def log(result: Any, logged_as: str) -> None:
            outputs.append({logged_as: result})

        def gen_uuid() -> str:
            return str(uuid4())[:18]

        def gen_user() -> dict[str, str]:
            return {"username": gen_uuid(), "password": gen_uuid()}

        def get_user(user: Users):
            return f"<Users> username: {user.username} | password: {user.password}"

        outputs = []
        random_users = [gen_user() for _ in range(4)]
        print("\n" * 10)
        try:
            # Create instances
            users = await Users.add_all(
                random_users[0], random_users[1], autocommit=True, refresh=True
            )
            log(
                [get_user(user) for user in users],
                "[add_all]",
            )
            user = await Users.add_one(autocommit=True, refresh=True, **random_users[2])
            log(get_user(user), "[add]")

            # Query instances
            stmt = select(Users).filter(
                Users.username.in_(
                    [random_users[0]["username"], random_users[1]["username"]]
                )
            )
            users = await db.scalars__fetchall(stmt)
            log(
                [get_user(user) for user in users],
                "[scalars__fetchall / in]",
            )
            result = await Users.update_where(
                random_users[0],
                {"username": "UPDATED", "password": "updated"},
                autocommit=True,
            )
            log(result, "[updated_where]")
            user = await Users.one_filtered_by(**random_users[2])
            log(get_user(user), "[one_filtered_by]")
            print("/" * 1000)
            user = await db.delete(user, autocommit=True)

            log(get_user(user), "[delete]")
            users = await Users.fetchall_filtered_by(**random_users[3])
            log(
                [get_user(user) for user in users],
                "[fetchall_filtered_by]",
            )
            stmt = select(Users).filter_by(**random_users[1])
            user = (await db.scalars(stmt=stmt)).first()
            log(
                get_user(user),
                "[sa.scalars().first()]",
            )

        except Exception as e:
            print("<" * 10, "Test failed!", ">" * 10)
            print("Detailed error:\n")
            raise e
        finally:
            await db.session.close()
            await db.engine.dispose()
            print("==" * 10, "Outputs", "==" * 10)
            for output in outputs:
                print(output, "\n")

    run(main())
