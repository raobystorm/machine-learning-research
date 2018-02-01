# python3

import os

labled_folder = '/Users/rui.zhong/nao_images'
res_file = labled_folder + '/result_nao.dat'

c_list = os.listdir('/Users/rui.zhong/nao_images/results/child')
f_list = os.listdir('/Users/rui.zhong/nao_images/results/father')
m_list = os.listdir('/Users/rui.zhong/nao_images/results/mother')
o_list = os.listdir('/Users/rui.zhong/nao_images/results/others')

with open(res_file, 'r') as f:
    strs = f.readlines()
    filename_list = strs[0].replace(' ', '').replace('\n', '').split(',')[:-1]

c_set = []
f_set = []
m_set = []
o_set = []

for img in filename_list:
    if img in c_list:
        c_set.append(img)
    elif img in f_list:
        f_set.append(img)
    elif img in m_list:
        m_set.append(img)
    else:
        o_set.append(img)


with open(labled_folder + '/results/labeled.txt', 'w') as f:
    f.write('Child images:\n')
    for img in c_list:
        f.write(img)
        f.write(', ')
    f.write('Father images:\n')
    for img in f_list:
        f.write(img)
        f.write(', ')
    f.write('Mother images:\n')
    for img in m_list:
        f.write(img)
        f.write(', ')
    f.write('Others images:\n')
    for img in o_list:
        f.write(img)
        f.write(', ')
