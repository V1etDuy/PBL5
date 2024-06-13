import socket
import cv2
import pickle
import struct
import threading
import torch
import pathlib
import time  # Importing the time module

pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = 'D:/TU LIEU DAI HOC/NAM 3/Ky 2/PBL5/CodeServer/best1.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 12345
socket_address = (host_ip, port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)

client_socket, addr = server_socket.accept()
print('GOT CONNECTION FROM:', addr)

data = b""
payload_size = struct.calcsize("Q")

def receive_and_process_frames():
    global data
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(1024) 
            if not packet:
                return
            data += packet
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(1024)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        if frame is not None:
            # Perform object detection
            results = model(frame)
            
            # Collect detection results
            detection_results = []
            for *box, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                detection_results.append(f"{label} - {conf:.2f}")
            
            # Join results into a single string
            message = "\n".join(detection_results)
            
            if message:
                # Send detection results to client
                client_socket.sendall(message.encode())
                print("Sent to client:", message)
        
        cv2.imshow('RECEIVING VIDEO', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Add delay (e.g., 0.5 seconds)
        time.sleep(0.5)

    client_socket.close()
    cv2.destroyAllWindows()

receive_thread = threading.Thread(target=receive_and_process_frames)
receive_thread.start()
receive_thread.join()
