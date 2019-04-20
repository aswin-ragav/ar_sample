import logging
from datetime import datetime

class Capture:

    CAPTURE = None

   # def __init__(self):
    #    self.CAPTURE =  instance()

    def Sketch(self, data = str : ""):
        capture_string = "- " + str(datetime.now()) + "  :- " + str(data)
        logging.info(capture_string + 'Start reading database')

def Init_Logger():
    logging.basicConfig(level=logging.DEBUG)
    #print(__name__) 
    logging.getLogger(__name__)

def Capture_Init():
    global CAPTURE
    CAPTURE = Capture()
    Init_Logger   

instance = Capture_Init
print(instance)
Capture.CAPTURE.Sketch("hiiiiiiiiiiiiiiii")