import asyncio

from dotenv import load_dotenv

load_dotenv()

from server import main  # noqa: E402

asyncio.run(main())
