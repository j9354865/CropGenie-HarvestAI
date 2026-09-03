import edge_tts, asyncio

async def test():
    communicate = edge_tts.Communicate(text="Hello world test", voice="en-US-JennyNeural")
    chunks = []
    async for chunk in communicate.stream():
        print("Event type:", chunk.get("type"), "| keys:", list(chunk.keys())[:5])
        chunks.append(chunk)
    audio = [c for c in chunks if c.get("type") == "audio"]
    print(f"\nAudio chunks: {len(audio)}, Total bytes: {sum(len(c.get('data', b'')) for c in audio)}")

asyncio.run(test())
