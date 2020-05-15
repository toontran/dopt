import asyncio


async def send_message(loop):
    reader, writer = await asyncio.open_connection("127.0.0.1", 15555)
    request = ""

    try:
        while request != "quit":
            request = input(">> ")
            if request:
                writer.write(request.encode("utf8"))
                response = (await reader.read(255)).decode("utf8")
                print(response, end="")
    except KeyboardInterrupt:
        writer.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(send_message(loop))
loop.close()