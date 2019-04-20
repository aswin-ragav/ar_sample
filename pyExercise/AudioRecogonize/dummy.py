'''import datetime
import InitMongoClient as imc
import json
from Description import Description

class dummy:

    def simple(self):
        desc = Description("Second Json insertion....!!!")
        #print(vars(desc))
        return desc

d = dummy()
ins = json.dumps(vars(d.simple()))
print(ins)
#desc.desc("FIrst Json insertion....!!!")
TO_DICT = {"0" : ins}
imc.create_Instance().insert_one(json.loads(ins)).inserted_id

#print(desc.desc + "  :-  " + datetime.datetime.now())
'''
import tensorflow as tf 

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])
alternates = tf.map_fn(lambda x: (x, x), matrix, dtype=(tf.int64, tf.int64))
print(alternates)
'''print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in matrix]))'''

