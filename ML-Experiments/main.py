import asyncio
from californiahousing.preprocessing import PreProcessingExperiment

async def main():
    exp = PreProcessingExperiment()
    await exp.run_async()

asyncio.run(main())



