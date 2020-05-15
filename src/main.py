import asyncio
from asyncio import sleep


async def handle_client(reader, writer):
    print("Start handler")
    request = None
    while request != "quit":
        request = (await reader.read(255)).decode("utf8")
        response = str(eval(request)) + '\n'
        writer.write(response.encode("utf8"))
        await writer.drain()
    writer.close()
    print("Close handler")
    loop.close()


async def side_task():
    print("Start")
    for i in range(100000):
        pass
    await sleep(10)
    print("End")


loop = asyncio.get_event_loop()
loop.create_task(
    asyncio.start_server(handle_client, "localhost", 15555)
)
loop.create_task(side_task())
loop.run_forever()
