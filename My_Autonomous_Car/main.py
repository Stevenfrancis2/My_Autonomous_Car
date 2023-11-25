import numpy as np
import cv2
import pandas as pd
import socket
import pygame
from datetime import datetime
import os
import multiprocessing
import queue



class VideoStreaming(object):
    def __init__(self, host, port, command):
        # connecting to socket
        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        #
        # initialize pygame
        pygame.font.init()
        WIDTH, HEIGHT = 320, 240
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NN car")

        self.lock = multiprocessing.Lock()
        self.command = command
        self.a = 0
        self.w = 0
        self.d = 0
        self.s = 0
        self.cmd = b'n'
        #######

        ####needed for directory
        self.newPath = ''
        self.mydir = ''
        self.countFolder = 0
        self.imglist = []
        self.steeringlist = []
        self.saveDataInit = False;

        ####constructor
        self.collect()

    def saveData(self, img, steering):
        now = datetime.now()
        timestamp = str(datetime.timestamp(now)).replace('.', '')
        filename = os.path.join(self.newPath, f'image_{timestamp}.jpg')
        cv2.imwrite(filename, img)
        self.imglist.append(filename)
        self.steeringlist.append(steering)

    def saveLog(self):

        rawData = {'Image': self.imglist,
                   'Steering': self.steeringlist}
        df = pd.DataFrame(rawData)
        df.to_csv(os.path.join(self.mydir, f'log_{str(self.countFolder)}.csv'), index=False, header=False)
        print('Log Saved')
        print('Total Images: ', len(self.imglist))

    def collect(self):
        self.mydir = os.path.join(os.getcwd(), 'DataCollected')
        while os.path.exists(os.path.join(self.mydir, f'IMG{str(self.countFolder)}')):
            self.countFolder += 1
        self.newPath = self.mydir + "\IMG" + str(self.countFolder)
        os.makedirs(self.newPath)
        '''
        self.mydir = os.path.join(os.getcwd(), 'DataCollected')

        # print(mydir)

        while os.path.exists(os.path.join(self.mydir, f'IMG{str(self.countFolder)}')):
            self.countFolder += 1
        newPath = self.mydir + "/IMG" + str(self.countFolder)
        os.makedirs(newPath)
        '''
        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")
            stream_bytes = b' '
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
                    ##### we have the image we  need the steering angle
                    cv2.imshow('image', image)
                    pygame.key.set_repeat(1, 333)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit()

                        if event.type == pygame.KEYDOWN:

                            if event.key == pygame.K_a:
                                self.a = 1

                            if event.key == pygame.K_d:
                                self.d = 1

                            if event.key == pygame.K_w:
                                self.w = 1

                            if event.key == pygame.K_s:
                                self.s = 1
                            if event.key == pygame.K_p:  ####press p to stop
                                self.saveLog()
                                self.saveDataInit = False;
                            if event.key == pygame.K_o:  #### press O to start
                                self.saveDataInit = True;
                        if event.type == pygame.KEYUP:

                            if event.key == pygame.K_a:
                                self.a = 0

                            if event.key == pygame.K_d:
                                self.d = 0

                            if event.key == pygame.K_w:
                                self.w = 0

                            if event.key == pygame.K_s:
                                self.s = 0
                    direction = [self.a, self.w, self.d, self.s]
                    ####here for each statment we are going to move the car
                    #### and save  the corresponding dir with the image

                    if direction == [1, 0, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'l')
                        self.lock.release()
                        self.cmd = b'l'
                        print("l")
                    elif direction == [0, 0, 1, 0]:
                        self.lock.acquire()
                        self.command.put(b'r')
                        self.lock.release()
                        self.cmd = b'r'
                        print("r")
                    elif direction == [0, 1, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'f')
                        self.lock.release()
                        self.cmd = b'f'
                        print("f")
                    elif direction == [0, 0, 0, 1]:
                        self.lock.acquire()
                        self.command.put(b'b')
                        self.lock.release()
                        self.cmd = b'b'
                        print("b")
                    elif direction == [0, 1, 1, 0]:
                        self.lock.acquire()
                        self.command.put(b'fr')
                        self.lock.release()
                        self.cmd = b'fr'
                        print("fr")
                    elif direction == [1, 1, 0, 0]:
                        self.lock.acquire()
                        self.command.put(b'fl')
                        self.lock.release()
                        self.cmd = b'fl'
                        print("fl")
                    elif direction == [1, 0, 0, 1]:
                        self.lock.acquire()
                        self.command.put(b'bl')
                        self.lock.release()
                        self.cmd = b'bl'
                        print("bl")
                    elif direction == [0, 0, 1, 1]:
                        self.lock.acquire()
                        self.command.put(b'br')
                        self.lock.release()
                        self.cmd = b'br'
                        print("br")

                    if self.saveDataInit:
                        self.saveData(image, self.cmd)

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
                                print(f'{mesg}  is poped out of the queue and will be sent to the raspberry')
                                self.connection.sendall(mesg)
                                action_taken = self.connection.recv(16)
                                print(f"the car is going {action_taken} ////ack received from the raspberry  ")


                            except queue.Empty:
                                self.connection.sendall(b'n')
                                action_taken = self.connection.recv(16)
                                print(f"the car is going {action_taken} ////ack received from the raspberry  ")



        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
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
