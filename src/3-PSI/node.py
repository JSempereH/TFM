from concurrent import futures
import grpc
import torch
import threading
import time
import protos.psi_pb2
import protos.psi_pb2_grpc

class PSIServicer(protos.psi_pb2_grpc.DataTransferServicer):
    def __init__(self):
        self.dataset = [] # Local dataset

    def SendData(self, request, context):
        self.dataset.extend(request.data)
        return protos.psi_pb2.DataResponse(message="Data received")
    
def serve(port, stop_event):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    protos.psi_pb2_grpc.add_DataTransferServicer_to_server(PSIServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started on port {port}")
    
    stop_event.wait()
    server.stop(0)
    print(f"Server on port {port} stopped")

def send_data(target_node, data):
    channel = grpc.insecure_channel(target_node)
    stub = protos.psi_pb2_grpc.DataTransferStub(channel)
    request = protos.psi_pb2.DataRequest(data=data)
    response = stub.SendData(request)
    print(f"Response from {target_node}: {response.message}")

def run_node(port, target_nodes):
    stop_event = threading.Event()

    # Initialize the server in a separate thread
    server_thread = threading.Thread(target=serve, args=(port,stop_event))
    server_thread.start()

    dataset = torch.Tensor([1.0, 2.0, 3.0])

    time.sleep(10)

    for target_node in target_nodes:
        data_to_send = dataset.tolist()
        print(data_to_send)
        send_data(target_node, data_to_send)

    stop_event.set()
    server_thread.join()

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: python node.py <port> <target_node_1> <target_node_2> ...")
        sys.exit()

    port = sys.argv[1]
    target_nodes = sys.argv[2:]

    run_node(port, target_nodes)
