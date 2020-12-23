# 载入运动数据帧（240*96），忽略骨骼信息
import re
import os
import torch
import random
import numpy as np

def loadBVHData(filename):

    with open(filename, 'r') as file:
        bvh = file.read()
        # 骨架信息
        skeleton = list(map(float, ' '.join(re.findall(r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
        skeleton.extend([0., 0., 0.])  # 补个零占位到 96，因为动作帧里开头有三个元素代表 xyz 坐标
        skeleton = [skeleton]  # [1, 96]
        #print(skeleton)


    with open(filename, 'r') as file:
        bvh2 = file.readlines()

        for i, content in enumerate(bvh2):
            """Frames是帧的总数."""
            if ('Frames:' in content):
                #print(content.split(':')[1])
                frameNumber = content.split(':')[1]
            """Frame Time用来算开始的索引."""
            if ('Frame Time' in content):
                #print(i)
                rowIndex = i


        """end代表需要截取的总帧数"""
        end = int((len(bvh2)-(rowIndex+1))/240)*240


        #frame里面装剪切后的若干个part
        sonframe = []
        frame = []
        label = int(os.path.basename(filename).split('_')[0]) - 1
        print("filename: " + os.path.basename(filename) + "  " + "label: " + str(label))

        if(int(frameNumber) < 240):
            print(os.path.basename(filename) + "  文件帧数不够！")
            return []
        else:
            i = 0
            while(i<end):
                for j in range(0, 240):
                    if(i<end):
                        #sonframe = []
                        if(len(bvh2[(rowIndex + 1) + i].split(' ')) != 96):
                            print(os.path.basename(filename) +  '这一行的长度是：' + str(len(bvh2[(rowIndex + 1) + i].split(' '))))
                        #print(len(bvh2[(rowIndex + 1) + i].split(' ')))
                        sonframe.append(list(map(float, bvh2[(rowIndex+1)+i].split(" "))))
                        i = i + 1
                    else:
                        break
                sondata = skeleton + list(sonframe)
                frame.append((sondata, label))
                sonframe.clear()
            #print(len(frame))
    return frame


def getAllData(dirpath, nums):
    data = []
    list = []
    """之前加载数据的方法，数据量太大，会出现内存错误"""
    # for root, dirs, files in os.walk(dirpath):
    #     for f in files:
    #         filename = os.path.join(root, f)
    #         #print(filename)
    #         if(loadBVHData(filename) != None):
    #             data.extend(loadBVHData(filename))

    for root, dirs, files in os.walk(dirpath):
        for f in files:
            filename = os.path.join(root, f)
            list.append(filename)
    #print(list)

    for i in range(nums):
        #print(list[i])
        data.extend(loadBVHData(list[i]))

    category_num = int(list[nums-1].split("\\")[-1].split("_")[0])
    print("category_num: " + str(category_num))
    return data, category_num


class MyDataset(torch.utils.data.Dataset):
    # rate 要使用的数据占全部的比率
    def __init__(self, dataset_dir='./data', rate=1.0):
        self.data, self.category_num = getAllData(dataset_dir, 20)
        # random.shuffle(self.data)
        # self.data = self.data[:int(len(self.data) * rate)]
        # exit()

        # 因为 id 从零开始，直接取最后一个文件的 id + 1
        #self.category_num = int(os.listdir(dataset_dir)[-1].split('_')[0]) + 1

    def __getitem__(self, index):
        data, label = self.data[index]
        return torch.FloatTensor(data), label

    def __len__(self):
        return len(self.data)




if __name__ == "__main__":

    # print('hello')
    data, category_num = getAllData('./data', 100)
    print(len(data))

    # frame = loadBVHData('01_01.bvh')
    # print(frame)

    # i = 0
    # for root, dirs, files in os.walk('./data'):
    #     for f in files:
    #         filename = os.path.join(root, f)
    #         i = i+1
    #         print(filename)
    # print(i)


    # dataset = MyDataset()
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=5, shuffle=True, drop_last=True)
    # for epoch in range(100):
    #     for batch_idx, (data, labels) in enumerate(dataloader):
    #         print(data)
    #         print(labels)
    #
    # print('hello')




