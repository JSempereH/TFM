import grpc
import asyncio
import psi_pb2
import psi_pb2_grpc
import numpy as np
from google.protobuf import empty_pb2
from threading import RLock

lock = asyncio.Lock()

async def prepare_for_psi(stub, column_index):
    async with lock:
        request = psi_pb2.PrepareForPSIRequest(index = column_index)
        await stub.PrepareForPSI(request) # The response is Empty

async def receive_encrypted_array(stub):
    async with lock:
        response = await stub.SendEncryptedArray(empty_pb2.Empty())
        encrypted_array = np.frombuffer(response.values, dtype=np.uint64)
        return encrypted_array

async def send_encrypted_array(stub, encrypted_array):
    async with lock:
        request = psi_pb2.ReceiveEncryptedArrayRequest(encrypted_values=encrypted_array.tobytes())
        await stub.ReceiveEncryptedArray(request)

async def get_pairs(stub):
    async with lock:
        response = await stub.GetPairs(empty_pb2.Empty())
        pairs = [(pair.y, pair.fy) for pair in response.pairs]
        return pairs
    
async def send_pairs(stub, pairs):
    async with lock:
        pair_messages = [psi_pb2.Pair(y=y, fy=fy) for y, fy in pairs]
        request = psi_pb2.SendPairsRequest(pairs=pair_messages)
        await stub.SendPairs(request)

async def compare_values(stub):
    async with lock:
        await stub.CompareValues(empty_pb2.Empty())

async def main():
    server_1_address = 'localhost:50051'
    server_2_address = 'localhost:50052'

    column_index = 0

    async with grpc.aio.insecure_channel(server_1_address) as channel1, \
               grpc.aio.insecure_channel(server_2_address) as channel2:
        stub1 = psi_pb2_grpc.DataTransferStub(channel1)
        stub2 = psi_pb2_grpc.DataTransferStub(channel2)

        # Prepares PSI: each server hashes, encrypts and sorts its data
        await asyncio.gather(
            prepare_for_psi(stub1, column_index),
            prepare_for_psi(stub2, column_index)
        )
    
        # Recibe los arrays encriptados de ambos servidores
        encrypted_array1, encrypted_array2 = await asyncio.gather(
            receive_encrypted_array(stub1),
            receive_encrypted_array(stub2)
        )

        print(f"Recibido de stub1: {encrypted_array1}")
        print(f"Recibido de stub2: {encrypted_array2}")

        # We send encrypted_array from stub1 to stub2 and viceversa:
        await asyncio.gather(
            send_encrypted_array(stub1, encrypted_array2),
            send_encrypted_array(stub2, encrypted_array1)
        )

        # Obtiene los pares (y, f(y)) de ambos servidores
        pairs1, pairs2 = await asyncio.gather(
            get_pairs(stub1),
            get_pairs(stub2)
        )

        # Envía los pares generados de stub1 a stub2 y viceversa
        await asyncio.gather(
            send_pairs(stub1, pairs2),
            send_pairs(stub2, pairs1)
        )

        # Muestra los pares recibidos de cada stub
        print(f"Pares enviados de stub1 a stub2: {pairs1}")
        print(f"Pares enviados de stub2 a stub1: {pairs2}")

        # Cada uno compara los pares
        await asyncio.gather(
            compare_values(stub1),
            compare_values(stub2)
        )


if __name__ == '__main__':
    asyncio.run(main())