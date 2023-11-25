import numpy as np
import cv2
import pandas as pd
import socket
import pygame
from datetime import datetime
import os
import multiprocessing
import queue
from tensorflow.keras.models import load_model



class VideoStreaming(object):
    def __init__(self, host, port, command,):
        # connecting to socket
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        #self.connection, self.client_address = self.server_socket.accept()
        #self.connection = self.connection.makefile('rb')
        #self.host_name = socket.gethostname()
        #self.host_ip = socket.gethostbyname(self.host_name)
        # to accept a single connection
        self.connection = self.server_socket.accept()[0].makefile('rb')
        self.model=load_model("C:\\Users\\jpg\\Desktop\\BEproject\\model.h5")
        #
        # initialize pygame
        pygame.font.init()
        WIDTH, HEIGHT = 320, 240
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NN car")

        self.lock = multiprocessing.Lock()
        self.command = command
        self.KeyPressFlag=False
        self.a = 0
        self.w = 0
        self.d = 0
        self.s = 0

        #######

        ####needed for directory


        ####constructor
        self.agent()

    def  getPredictedDirection(self,RawStreering):
        MaxIndex = np.argmax(RawStreering)
        if MaxIndex==0:
            print("forward left predicted")
            return [1,1,0,0]
        elif MaxIndex==1:
            print("forward predicted")
            return [0,1,0,0]
        elif MaxIndex==2:
            print("forward right predicted")
            return [0,1,1,0]


    def agent(self):

        stream_bytes = b' '

        try:
            # print("Host: ", self.host_name + ' ' + self.host_ip)
            # print("Connection from: ", self.client_address)
            # print("Streaming...")
            # print("Press 'q' to exit")

            self.lasttime = 0
            self.lasterror = 0

            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('Streaming', image)

                    #### predictions are made  here ####

                    ######we have to preprocess the image to fit the model
                    #image =cv2.resize(image,dsize=(480,240))
                    #image = np.asarray(image)
                    #image = image[54:120, :, :]
                    #image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    #image = cv2.GaussianBlur(image, (3, 3), 0)
                    image = cv2.resize(image, (200, 66))
                    image = image / 255
                    image = np.array([image])

                    ##################
                    RawSteering = self.model.predict(image)

                    # print("model prediction")
                    # print(RawSteering)


                    pygame.key.set_repeat(1, 333)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit()

                        if event.type == pygame.KEYDOWN:
                            self.KeyPressFlag=True

                            if event.key == pygame.K_a:
                                self.a = 1

                            if event.key == pygame.K_d:
                                self.d = 1

                            if event.key == pygame.K_w:
                                self.w = 1

                            if event.key == pygame.K_s:
                                self.s = 1

                        if event.type == pygame.KEYUP:
                            self.KeyPressFlag = False
                            if event.key == pygame.K_a:
                                self.a = 0

                            if event.key == pygame.K_d:
                                self.d = 0

                            if event.key == pygame.K_w:
                                self.w = 0

                            if event.key == pygame.K_s:
                                self.s = 0

                    # if there is a keyboard input ignore the prediction
                    # if there is no keyboard input take the direction predicted
                    direction = [self.a, self.w, self.d, self.s]
                    if self.KeyPressFlag:

                        print("keyboard input")
                        direction = [self.a, self.w, self.d, self.s]
                    else:
                        pass
                        #direction=self.getPredictedDirection(RawSteering)

                    ####here for each statment we are going to move the car
                    ####
                    # left
                    if direction == [1, 0, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'l')
                        self.lock.release()
                    # right
                    elif direction == [0, 0, 1, 0]:
                        self.lock.acquire()
                        self.command.put(b'r')
                        self.lock.release()
                    # forward
                    elif direction == [0, 1, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'f')
                        self.lock.release()
                    # backward
                    elif direction == [0, 0, 0, 1]:
                        self.lock.acquire()
                        self.command.put(b'b')
                        self.lock.release()
                    # forward right
                    elif direction == [0, 1, 1, 0]:
                        self.lock.acquire()
                        self.command.put(b'fr')
                        self.lock.release()
                    # forward Left
                    elif direction == [1, 1, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'fl')
                        self.lock.release()
                    # backward left
                    elif direction == [1, 0, 0, 1]:
                        self.lock.acquire()
                        self.command.put(b'bl')
                        self.lock.release()
                    # backward right
                    elif direction == [0, 0, 1, 1]:
                        self.lock.acquire()
                        self.command.put(b'br')
                        self.lock.release()


        finally:
            self.connection.close()
            self.server_socket.close()


class SendCommand(object):

    def __init__(self, host, port, command):

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.cmd = command

        self.lock = multiprocessing.Lock()
        # self.host_name = socket.gethostname()
        # self.host_ip = socket.gethostbyname(self.host_name)
        self.sendCOMMAND()

    def sendCOMMAND(self):

        try:
            # print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Sending command..... ")
            print("Press 'q' to exit")
            while True:
                while True:
                    while True:
                        while True:
                            try:

                                # self.lock.acquire()
                                mesg = self.cmd.get(True, 0.240)
                                # self.lock.release()
                                #print(f'{mesg}  is poped out of the queue and will be sent to the raspberry')
                                self.connection.sendall(mesg)
                                action_taken = self.connection.recv(16)
                                #print(f"the car is going {action_taken} ////ack received from the raspberry  ")


                            except queue.Empty:
                                self.connection.sendall(b'n')
                                action_taken = self.connection.recv(16)
                                #print(f"the car is going {action_taken} ////ack received from the raspberry  ")



        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # host, port
    command = multiprocessing.Queue()
    h1, p1 = "0.0.0.0", 777
    h2, p2 = "0.0.0.0", 65432



    p1 = multiprocessing.Process(target=VideoStreaming, args=(h1, p1, command,))
    p2 = multiprocessing.Process(target=SendCommand, args=(h2, p2, command,))
    p1.start()
    p2.start()

    p2.join()
    p1.join()
    print('We  are  all    Done')
