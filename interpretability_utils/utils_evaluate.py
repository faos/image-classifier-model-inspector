import numpy as np


def full_origins(x_sz, y_sz, size=5):
    origins = []
    for i in range(0, x_sz, size):
        for j in range(0, y_sz, size):
            x_end, y_end = i + size, j + size
            if x_end >= x_sz or y_end >= y_sz: continue
            origins.append((i, j))
    return origins


def all_origins(n, x_sz, y_sz, size=5):
    origins = []
    x_list = np.random.randint(low=0, high=x_sz, size=n)
    y_list = np.random.randint(low=0, high=y_sz, size=n)
    for i in range(x_sz):
        for j in range(y_sz):
            x_end, y_end = i + size, j + size
            if x_end >= x_sz or y_end >= y_sz: continue
            origins.append((i, j))
    return origins


def random_origins(n, x_sz, y_sz, size=5):
    origins = []
    x_list = np.random.randint(low=0, high=x_sz, size=n)
    y_list = np.random.randint(low=0, high=y_sz, size=n)
    for i in range(x_list.shape[0]):
        x_end, y_end = x_list[i] + size, y_list[i] + size
        if x_end >= x_sz or y_end >= y_sz: continue
        origins.append((x_list[i], y_list[i]))
    return origins


def delete_region(img, size, x, y, gaussiana=False):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    x_sz, y_sz = img.shape[2], img.shape[3]
    x_end, y_end = x + size, y + size
    #print('sum before: ', img.sum())
    for i in range(x, x_end):
        for j in range(y, y_end):
            if i >= x_sz or j >= y_sz: continue
            if gaussiana:
                img[0][0][i][j] = float(np.random.normal(0.0, 1.0, 1))
                img[0][1][i][j] = float(np.random.normal(0.0, 1.0, 1))
                img[0][2][i][j] = float(np.random.normal(0.0, 1.0, 1))
            else:
                img[0][0][i][j] = 0.0
                img[0][1][i][j] = 0.0
                img[0][2][i][j] = 0.0
    #print('sum after: ', img.sum())
    return img


def delete_region2d(img, size, x, y, gaussiana=False):
    x_sz, y_sz = img.shape[1], img.shape[2]
    x_end, y_end = x + size, y + size
    #print('sum before: ', img.sum())
    for i in range(x, x_end):
        for j in range(y, y_end):
            if i >= x_sz or j >= y_sz: continue
            img[0][i][j] = 0.0
            img[0][i][j] = 0.0
            img[0][i][j] = 0.0
    #print('sum after: ', img.sum())
    return img


def sum_matrix(matrix):
    sum = np.zeros(matrix.shape, dtype=np.float32)
    w, h = matrix.shape[0], matrix.shape[1]
    
    for i in range(w):
        for j in range(h):
            sum[i][j] = matrix[i][j]
            if i > 0: sum[i][j] += sum[i - 1][j]
            if j > 0: sum[i][j] += sum[i][j - 1]
            if i > 0 and j > 0: sum[i][j] -= sum[i - 1][j - 1]
            
    """
    consultar de i,j ate k,l
    k, l = i + width, j + width
    if k >= w or l >= h: continue 
    sub = sum[k][l]
    if i > 0: sub -= sum[i - 1][l]
    if j > 0: sub -= sum[k][j - 1]
    if i > 0 and j > 0: sub += sum[i - 1][j - 1]
    """

    return sum


def generate_imgs(img, size, origins, is2d=True):
    imgs = []
    for x, y, _ in origins:
        if is2d:
            imgs.append(delete_region2d(np.copy(img), size, x, y))
        else:
            imgs.append(delete_region(np.copy(img), size, x, y))
        
    return imgs


def generate_imgs_from_origins(img, size, origins, is2d=True, gaussiana=True):
    imgs = []
    for x, y in origins:
        if is2d:
            imgs.append(delete_region2d(np.copy(img), size, x, y, gaussiana))
        else:
            imgs.append(delete_region(np.copy(img), size, x, y, gaussiana))
    #print('generate imgs from origins:', len(imgs))
    return imgs


def morf(img, size, origins, attribution, gaussiana=True):
    sum_att = sum_matrix(attribution)
    seq = build_importances(origins, size, sum_att)[::-1]
    imgs = [np.copy(img)]
    pos = [(-1, -1)]
    for x, y, sub in seq:
        imgs.append(delete_region(np.copy(imgs[-1]), size, x, y, gaussiana))
        pos.append((x, y))
    return imgs, pos

def build_importances(origins, size, matrix_sum):
    #print("shape of m sum:", matrix_sum.shape)
    x_sz, y_sz = matrix_sum.shape
    output = []
    """
    sub = sum[k][l]
    if i > 0: sub -= sum[i - 1][l]
    if j > 0: sub -= sum[k][j - 1]
    if i > 0 and j > 0: sub += sum[i - 1][j - 1]
    """
    for x, y in origins:
        x_end, y_end = x + size, y + size
        if x_end >= x_sz: x_end = x_sz - 1
        if y_end >= y_sz: y_end = y_sz - 1
        sub = matrix_sum[x_end][y_end]
        if x > 0: sub -= matrix_sum[x - 1][y_end]
        if y > 0: sub -= matrix_sum[x_end][y - 1]
        if x > 0 and y > 0: sub += matrix_sum[x - 1][y - 1] 
        output.append([x, y, sub])
    output.sort(key=lambda tup: tup[2])
    return output

def build_seq_importance(imgs, seq_pos, size, preds, matrix_sum):
    x_sz, y_sz = matrix_sum.shape
    output = []
    for i in range(len(seq_pos)):
        x, y = seq_pos[i]
        x_end, y_end = x + size, y + size
        if x_end >= x_sz: x_end = x_sz - 1
        if y_end >= y_sz: y_end = y_sz - 1
        sub = matrix_sum[x_end][y_end]
        if x > 0: sub -= matrix_sum[x - 1][y_end]
        if y > 0: sub -= matrix_sum[x_end][y - 1]
        if x > 0 and y > 0: sub += matrix_sum[x - 1][y - 1] 
        output.append([imgs[i], x, y, sub, preds[i]])
    output.sort(key=lambda tup: tup[3])
    return output
        


if __name__ == "__main__":
    origins = random_origins(40, x_sz=100, y_sz=100)