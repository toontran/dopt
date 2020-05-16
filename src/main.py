import asyncio
from asyncio import sleep
import time


l = [0]*100000000


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    print("Start handler")
    request = None
    while request != "quit":
        request = (await reader.read(255)).decode("utf8")
        response = str(eval(request)) + '\n'
        writer.write(response.encode("utf8"))
        await writer.drain()
    writer.close()
    print(f"Closing Trainer at {writer.get_extra_info('peername')}")


async def side_task():
    print("Start")
    await sleep(10)
    print("End")


# loop = asyncio.get_event_loop()
# loop.create_task(
#     asyncio.start_server(handle_client, "localhost", 15555)
# )
# loop.create_task(side_task())
# loop.run_forever()

async def main():
    server = await asyncio.start_server(handle_client,
                                        "localhost", 15555)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    await side_task()

    async with server:
        await server.serve_forever()
        print("servered")


asyncio.run(main())

