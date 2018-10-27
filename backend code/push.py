import json
import urllib
import urllib.request

import serial

firebase_url = 'https://qvik-b52d7.firebaseio.com/'
# Connect to Serial Port for communication
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0)


def update_entry(data_c, no):
    try:
        my_data = dict()
        data_c = data_c[0:len(data_c) - 4]
        my_data = {'value': (data_c)}

        json_data = json.dumps(my_data).encode()
        request = urllib.request.Request("https://qvik-b52d7.firebaseio.com//" + str(1) + ".json", data=json_data,
                                         method="PATCH")
        loader = urllib.request.urlopen(request)
        request1 = urllib.request.Request("https://qvik-b52d7.firebaseio.com//" + no + ".json", data=json_data,
                                          method="PATCH")
        loader = urllib.request.urlopen(request1)
    except urllib.error.URLError as e:
        message = json.loads(e.read())
        print(message["error"])
    else:
        print(loader.read())


i = 0
while 1:
    i = i + 1
    data_c = str(ser.readline())
    update_entry(data_c, str(i))
