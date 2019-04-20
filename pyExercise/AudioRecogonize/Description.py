import datetime
import random as random
class Description:

    def __init__(self, BRIEF):
        self.id = random.randrange(1000)
        self.timestamp =  datetime.datetime.now().strftime('%Y_%m_%d')
        self.desc = BRIEF
