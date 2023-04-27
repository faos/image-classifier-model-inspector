import numpy as np
import pandas as pd
import torch
import torchvision
import random

from interpretability_utils.utils_evaluate import *
from model_utils.inference import BatchPredictorFromTensor

def has_u(predictions):
    pred_max = np.array(predictions).max()
    high = [False]*len(predictions)
    low = [False]*len(predictions)
    first_high, last_high = -1, -1

    for i in range(len(predictions)):
        p = predictions[i]
        
        if p >= 0.8*pred_max:
            high[i] = True

        if p < 0.8*pred_max:
            low[i] = True

    for i in range(len(high)):
        if first_high == -1 and high[i] == True:
            first_high = i
        if high[i] == True:
            last_high = i
    
    return True in low[first_high:last_high + 1], last_high

def get_u_positions(predictions, target):
    y_pred = predictions.argmax(1)
    pos_pred_correct = np.where(y_pred == target)[0].tolist()
    if len(pos_pred_correct) >= 2:
        u_triples = []
        for i in range(len(pos_pred_correct) - 1):
            idx_left = pos_pred_correct[i]
            idx_right = pos_pred_correct[i + 1]
            if (idx_right - idx_left) > 1: #there is a gap, at least one image between them is different than target
                middle = random.choice(list(range(idx_left + 1, idx_right)))
                u_triples.append((idx_left, middle, idx_right))
        return u_triples
    else:
        return []



def noise_region_mean(img, seq_pos, size, cumulative):
    ans = [img.copy()]

    for p in range(0, len(seq_pos)):
        (i, j), importance = seq_pos[p]
        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()
        newimg[0, 0, i:i + size, j:j + size] = img[0, 0, i:i + size, j:j + size].mean()
        newimg[0, 1, i:i + size, j:j + size] = img[0, 1, i:i + size, j:j + size].mean()
        newimg[0, 2, i:i + size, j:j + size] = img[0, 2, i:i + size, j:j + size].mean()
        ans.append(newimg.astype(np.float32))
        
    return ans


def noise_img_mean(img, seq_pos, size, cumulative):
    img_mean = [img[0, 0, :, :].mean(), img[0, 1, :, :].mean(), img[0, 2, :, :].mean()]
    ans = [img.copy()]

    for p in range(0, len(seq_pos)):
        (i, j), importance = seq_pos[p]
        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()
        newimg[0, 0, i:i + size, j:j + size] = img_mean[0]
        newimg[0, 1, i:i + size, j:j + size] = img_mean[1]
        newimg[0, 2, i:i + size, j:j + size] = img_mean[2]
        ans.append(newimg.astype(np.float32))
        
    return ans


def noise_mean_norm(img, seq_pos, size, cumulative):
    mean = np.array([0.4914, 0.4822, 0.4465])
    ans = [img.copy()]
    for p in range(0, len(seq_pos)):
        (x, y), importance = seq_pos[p]
        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()
        for i in range(x, x + size):
            for j in range(y, y + size):
                if i < 0 or i >= 32 or j < 0 or j >= 32: continue
                newimg[0][0][i][j] = mean[0]
                newimg[0][1][i][j] = mean[1]
                newimg[0][2][i][j] = mean[2]
        ans.append(newimg.astype(np.float32))
    return ans


def noise_black(img, seq_pos, size, cumulative):
    ans = [img.copy()]
    for p in range(0, len(seq_pos)):
        (i, j), importance = seq_pos[p]
        
        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()

        newimg[0, 0, i:i + size, j:j + size] = 0.0
        newimg[0, 1, i:i + size, j:j + size] = 0.0
        newimg[0, 2, i:i + size, j:j + size] = 0.0

        ans.append(newimg.astype(np.float32))
        
    return ans


def noise_white(img, seq_pos, size, cumulative):
    ans = [img.copy()]
    for p in range(0, len(seq_pos)):
        (i, j), importance = seq_pos[p]

        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()

        newimg[0, 0, i:i + size, j:j + size] = 1.0
        newimg[0, 1, i:i + size, j:j + size] = 1.0
        newimg[0, 2, i:i + size, j:j + size] = 1.0

        ans.append(newimg.astype(np.float32))
        
    return ans


def gaussian_noise(img, seq_pos, size, cumulative):
    mean = np.array([0.4914, 0.4822, 0.4465])
    ans = [img.copy()]

    for p in range(0, len(seq_pos)):
        (x, y), importance = seq_pos[p]

        if cumulative:
            newimg = ans[-1].copy()
        else:
            newimg = img.copy()

        for i in range(x, x + size):
            for j in range(y, y + size):
                #if i < 0 or i >= 32 or j < 0 or j >= 32: continue
                try:
                    newimg[0][0][i][j] = float(np.random.normal(0.0, 1.0, 1))
                    newimg[0][1][i][j] = float(np.random.normal(0.0, 1.0, 1))
                    newimg[0][2][i][j] = float(np.random.normal(0.0, 1.0, 1))
                except:
                    pass

        ans.append(newimg.astype(np.float32))

    return ans


def noise(noise_method, img, seq_pos, size, cumulative):

    if noise_method == 'gaussian':
        ans = gaussian_noise(img, seq_pos, size, cumulative)
    elif noise_method == 'black':
        ans = noise_black(img, seq_pos, size, cumulative)
    elif noise_method == 'mean_norm':
        ans = noise_mean_norm(img, seq_pos, size, cumulative)
    elif noise_method == 'img_mean':
        ans = noise_img_mean(img, seq_pos, size, cumulative)
    elif noise_method == 'region_mean':
        ans = noise_region_mean(img, seq_pos, size, cumulative)
    elif noise_method == 'white':
        ans = noise_white(img, seq_pos, size, cumulative)

    return ans


def get_pos_importance(pos, size, matrix_sum):
    x_sz, y_sz = matrix_sum.shape
    x, y = pos
    
    x_end, y_end = x + size, y + size

    if x_end >= x_sz: x_end = x_sz - 1
    if y_end >= y_sz: y_end = y_sz - 1

    sub = matrix_sum[x_end][y_end]

    if x > 0: sub -= matrix_sum[x - 1][y_end]
    if y > 0: sub -= matrix_sum[x_end][y - 1]
    if x > 0 and y > 0: sub += matrix_sum[x - 1][y - 1] 
    
    return sub


def generate_origins(size, height=224, width=224):
    ans = []
    for i in range(0, height, size):
        for j in range(0, width, size):
            #if i + size >= height or j + size >= width: continue
            ans.append((i, j))
    return ans


def generate_only_sorted_positions(matrix_sum, size, reverse_mode=True):
    pos = generate_origins(size)
    ans_full = []
    ans_pos = []
    ans_imp = []

    for x, y in pos:
        ans_full.append(((x, y), get_pos_importance((x, y), size, matrix_sum)))

    ans_full.sort(key=lambda tup:tup[1], reverse=reverse_mode)

    for (x, y), imp in ans_full:
        ans_pos.append((x, y))
        ans_imp.append(imp)

    return ans_pos, ans_imp


def strategy0(attribution, size):
    """
    Dado a interpretação do target com relação a entrada, 
    retorna as posições mais importantes ordenadas de forma DECRESCENTE.
    """
    
    att = attribution.squeeze()
    att_sum = sum_matrix(att)
    pos, imps = generate_only_sorted_positions(att_sum, size, reverse_mode=True)

    return pos, imps


def strategy_crescent(attribution, size):
    """ 
    Dado a interpretação do target com relação a entrada, 
    retorna as posições mais importantes ordenadas de forma CRESCENTE.
    """
    
    att = attribution.squeeze()
    att_sum = sum_matrix(att)
    pos, imps = generate_only_sorted_positions(att_sum, size, reverse_mode=False)

    return pos, imps 


def get_gen_batch(x, y, att, order, window_size, noise_method, cumulative):
    assert len(att.shape) == 2
    
    if order == 'crescent':
        ##calcula a importância de cada região de dimensão window_size e retorna as origens de forma ordenada (crescent)
        pos_sorted, imp_sorted = strategy_crescent(att, size=window_size)
        pos_sub_sorted = list(zip(pos_sorted, imp_sorted))

    elif order == 'decrescent':
        ##calcula a importância de cada região de dimensão window_size e retorna as origens de forma ordenada (decrescente)
        pos_sorted, imp_sorted = strategy0(att, size=window_size)
        pos_sub_sorted = list(zip(pos_sorted, imp_sorted))
        
    elif order == 'random':
        ##calcula a importância de cada região de dimensão window_size e retorna as origens de forma ordenada (decrescente)
        ## em seguida executa um shuffle na ordem gerada
        pos_sorted, imp_sorted = strategy0(att, size=window_size)
        pos_sub_sorted = list(zip(pos_sorted, imp_sorted))
        random.shuffle(pos_sub_sorted)
    
    print('Length of pos_sub_sorted:', len(pos_sub_sorted))

    ##gera as novas imagens "borradas" de acordo com o método de ruído escolhido
    newx = torch.from_numpy(np.asarray(x)).permute(2, 0, 1).unsqueeze(0).numpy() / 255
    print("newx shape:", newx.shape)

    gen_x = noise(noise_method, newx, pos_sub_sorted, window_size, cumulative)

    gen_batch = torch.from_numpy(np.concatenate(gen_x, 0).astype(np.float32))
    
    return gen_batch, pos_sorted, imp_sorted

def build_region_importance_heatmap(input_shape, window_size, pos_sorted, imp_sorted):
    heatmap = np.zeros((1, input_shape[0], input_shape[1]))
    
    for pos in range(len(pos_sorted)):
        x, y = pos_sorted[pos]
        heatmap[:, x:x+window_size, y:y+window_size] = imp_sorted[pos]
    
    return heatmap


def analysis(model, x, y, att, order, window_size, noise_method, cumulative):
    print('\nanalysis'.upper())
    gen_batch, pos_sorted, imp_sorted = get_gen_batch(
        x, y, att, order, window_size, 
        noise_method, cumulative
        )
    print("\n\nGEN BATCH SHAPE:", gen_batch.shape, gen_batch[0].shape, gen_batch[0].max(), gen_batch[-1].min(), "\n\n")
    print("\n\nGEN BATCH SHAPE:", gen_batch.shape, gen_batch[-1].shape, gen_batch[-1].max(), gen_batch[-1].min(), "\n\n")

    output = BatchPredictorFromTensor()(model, gen_batch)
    
    heatmap = build_region_importance_heatmap((224, 224), window_size, pos_sorted, imp_sorted)
    for pos in range(len(pos_sorted)):
        print(pos_sorted[pos], imp_sorted[pos])

    print(output.shape)
    print(output[:, y])
    print(output.argmax(1))
    
    
    return output, heatmap, gen_batch


    