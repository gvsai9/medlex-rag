import asyncio
from state_tracker import get_tracker

async def main():
    tracker = get_tracker()
    await tracker.init_db()
    stats = await tracker.get_stats()
    print("MySQL connected. Stats:", stats)
    await tracker.close()

asyncio.run(main())