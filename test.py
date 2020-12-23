import datetime
import os
import torch
import numpy as np
#print(torch.__version__)



# print(datetime.datetime.now().day)
# print(datetime.datetime.now().hour)
# print(datetime.datetime.now().minute)
# a = datetime.datetime.now().d
# b = datetime.datetime.now().hour
# c = datetime.datetime.now().minute
#
#
# string = str(a) + '-' + str(b) + ':' + str(c)
# print(string)

# string = str(datetime.datetime.now().day) + '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute)
# os.mkdir('./models/'+string)
# for root, dirs, files in os.walk('./data'):
#     for f in files:
#         print(os.path.join(root, f))



# a = [([1,2],[3,4],2)]
# b = [([1,2],[3,4],2)]
#
# #a.extend(b)
# print(a+b)



# a = '1 2 3 4 5 6'
# #b = list(a.replace(' ', ',')).remove("1")
# b = a.split(" ")
# b = list(map(float, a.split(" ")))
#
# print(b)




# list = [1,2,2,2]
# list.extend([])
# print(list)

# list = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 2, 3], [4, 9, 6], [7, 8, 9]]]
# list = np.array(list)
# print(list.shape)
# list2 = np.concatenate(list, axis=1)
# print(list2)
# print(list[:, 1, :])
# print(list[:, -1, 0])


# st = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# print(st[2:6:2])
# print(st[6:2:-2])
# print(st[::1])
# print(st[::-1])  # 倒序输出
# print(st[-1])
# date = str(datetime.datetime.now().day) + '-' + str(datetime.datetime.now().hour) + '-' + str(datetime.datetime.now().minute) + '-' + str(datetime.datetime.now().second)
# print(date)
for i in range(239, 0, -1):
    print(i)