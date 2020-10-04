import pandas as pd
import os
import glob
import numpy as np




train_img = glob.glob("data/*.*")

train_mask = []

for i in train_img:
    z = i.replace('data','mask')
    x = z.replace('jpg','png')
    train_mask.append(x)


train_list = pd.DataFrame(np.concatenate([np.asarray(train_img).reshape(-1,1),np.asarray(train_mask).reshape(-1,1)],axis=1))


train_list.to_csv('train.csv',index=False)

# test_img = glob.glob("data/test/*.*")

# test_mask = []

# for i in test_img:
#     z = i.replace('data','mask')
#     x = z.replace('jpg','png')
#     test_mask.append(x)


# test_list = pd.DataFrame(np.concatenate([np.asarray(test_img).reshape(-1,1),np.asarray(test_mask).reshape(-1,1)],axis=1))


# test_list.to_csv('test.csv',index=False)

