import asyncio
import json
import random

# 本机地址
HOST = '127.0.0.1'
PORT = 9760


async def handle_client(reader, writer):
    print(f"✅ GUI 已连接到虚拟机械臂")

    try:
        while True:
            # 1. 接收 GUI 指令
            try:
                data = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if data:
                    msg = data.decode('utf-8')
                    print(f"📥 [虚拟臂] 收到: {msg}")

                    if "AddPoints" in msg:
                        print("   -> [虚拟臂] 回复: 1 (接收成功)")
                        writer.write("1".encode('utf-8'))
                        await writer.drain()
            except asyncio.TimeoutError:
                pass

                # 2. 模拟机械臂主动发“拍照” (按需触发)
            await asyncio.sleep(0.1)
            # 1% 的概率随机触发拍照，模拟现场信号
            if random.random() < 0.01:
                cmd = {"reqType": "photo", "dsID": "cam", "camID": "0"}
                print("📤 [虚拟臂] 发送: 拍照请求")
                writer.write(json.dumps(cmd).encode('utf-8'))
                await writer.drain()

    except Exception as e:
        print(f"❌ 连接断开: {e}")
    finally:
        writer.close()


async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"🤖 虚拟机械臂运行中... {HOST}:{PORT}")
    await server.serve_forever()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass