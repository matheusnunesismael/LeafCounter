import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import copy
import time
from scipy import ndimage
import os, sys

WHITE = 240 #constante para checagem de branco

#dicionário que define as 8 posíveis posições de um vizinho, iniciando pelo vizinho à esquerda
neighbors = {
    "n_0": [ 0,-1],
    "n_1": [-1,-1],
    "n_2": [-1, 0],
    "n_3": [-1, 1],
    "n_4": [ 0, 1],
    "n_5": [ 1, 1],
    "n_6": [ 1, 0],
    "n_7": [ 1,-1]
}

inverse_neighbors = {
    "[-1 -1]": 1,
    "[-1  0]": 2,
    "[-1  1]": 3,
    "[ 0 -1]": 0,
    "[0 1]"  : 4,
    "[ 1 -1]": 7,
    "[1 0]"  : 6,
    "[1 1]"  : 5
}

freeman_dict = ['4', '3', '2', '1', '0', '7', '6', '5']

# função para buscar no dicionário em qual posição "neig" está em relação a "center"
def find_n(center, neig):
    delta = neig - center
    cont_n = inverse_neighbors[str(delta)]

    return cont_n

# função para verificar se o ponto p está dentro dos limites da matriz
def in_limits(p, limits):
    return p[0] < limits[0] and p[0] >= 0 and p[1] < limits[1] and p[1] >= 0

# rotaciona a sequencia de freeman para obter o menor valor inteiro
def rot_freeman(freeman):
    free_min_str = freeman
    free_min = int(freeman)
    for i in range(len(freeman)):
        freman_len_minus_1 = len(freeman)-1
        new_freeman = freeman[freman_len_minus_1] + freeman[:freman_len_minus_1]
        if(int(new_freeman) < free_min):
            free_min = int(new_freeman)
            free_min_str = new_freeman
        freeman = new_freeman
    return free_min_str

# pega a sequencia de diferenças no caminho da sequencia de freeman
def first_dif(freeman):
    freman_len = len(freeman)
    first_dif_str = str((int(freeman[0]) - int(freeman[freman_len-1]))%8)
    for i in range(1, freman_len)):
        first_dif_str += str((int(freeman[i]) - int(freeman[i-1]))%8)
    return first_dif_str


# passos 3 a 5 do Agoritmo Seguidor de Fronteira
def frontier_explorator(b, c, matrix, b_0, frontier, freeman):
    height = np.shape(matrix)[0]
    width  = np.shape(matrix)[1]
    # flag booleana que determina se o laço deve continuar. Torna-se verdadeira quando b=b_0
    back_to_beginning = False
    border = False
    freeman_cont = 0
    while not back_to_beginning:
        # verifica qual o n do c atual
        cont_n = find_n(b, c)
        
        # incrementa n para começar a olhar para o próximo vizinho
        cont_n = (cont_n+1)%8

        # se o proximo vizinho está fora dos limites da imagem, caminha pelos vizinhos para encontrar um contido na imagem
        while not in_limits(b + neighbors["n_"+str(cont_n)], [height, width]):
            cont_n = (cont_n+1)%8

        # nk = próximo vizinho
        nk = b + neighbors["n_"+str(cont_n)]

        # laço busca o próximo pixel vizinho que não é branco
        while np.average(matrix[nk[0], nk[1]])>WHITE:
            cont_n = (cont_n+1)%8
            nk = b + neighbors["n_"+str(cont_n)]

        # adiciona a posição do proximo pixel a sequencia de freeman
        freeman += freeman_dict[cont_n]
        
        # cria a variável "k_minus_1", que guarda o índice do vizinho anterior
        k_minus_1 = (cont_n-1)%8

        # c corresponde ao vizinho anterior, b corresponde ao vizinho de valor 1
        c = b + neighbors["n_"+str(k_minus_1)]
        b = nk

        # joga o valor de b pra lista de pixels da fronteira
        frontier += [b]

        # verifica se b já voltou pro começo pra continuar a percorrer a fronteira
        if np.array_equal(b,b_0):
            back_to_beginning = True
    return (frontier, freeman)

# inicia o Algoritmo Seguidor de Fronteira com os Passos 1 e 2
def frontier_finder(last_b_0, matrix):
    all_done = False
    freeman = ""
    folha = True

    # percorre os pixels brancos da imagem até encontrar o primeiro pixel não-branco
    avg = np.mean(matrix,axis=-1)
    nao_nulo = avg<240
    achou = np.transpose(np.nonzero(nao_nulo))
    if len(achou)<20:
        return ([], True, last_b_0, folha, freeman)
    while achou[0,0] == 0:
        achou = np.delete(achou,0,0)
    b_0 = achou[0,:]

    # define c como o vizinho da esquerda do primeiro pixel não-branco
    c = b_0 + neighbors["n_1"]

    # procura o primeiro vizinho não branco para ser o próximo c
    cont_n = 1  #testar se cont_n=0 não funciona
    while np.average(matrix[c[0], c[1]])>WHITE:
        c = b_0 + neighbors["n_"+str(cont_n)]
        cont_n = (cont_n+1)%8
        if cont_n==0:
            achou = np.delete(achou,0,0)
            if len(achou)==0:
                return ([], True, b_0, folha, freeman)
            else:
                b_0 = achou[0,:]
                c = b_0 + neighbors["n_0"]
                cont_n = 0

    # adiciona a primeira mudança de direção à sequencia de freeman
    freeman += freeman_dict[cont_n]           
   
    # passa o valor do atual pixel não-branco (c) para b, e passa o valor de b_0 para c
    b, c = c, b_0
    
    # adiciona b_0 e b para a lista de pixels percorridos
    frontier = [b_0, b] 
    frontier_resp, freeman_resp = frontier_explorator(b, c, matrix, b_0, frontier, freeman)
    frontier += frontier_resp
    freeman += freeman_resp

    if len(frontier) < 50 or len(frontier) >= 19500:
        folha = False

    return (frontier, all_done, b_0, folha, freeman)

def segmentation(img, last_b_0):
    # chama o algoritmo seguidor de fronteira e armazena sua fronteira em "frontier"
    (frontier, all_done, last_b_0, folha, freeman) = frontier_finder(last_b_0, img)
    
    if not all_done:
        frontier_matrix = np.array(frontier)

        # define dimensões da sub-imagem
        min_y = np.min(frontier_matrix[:,0])
        max_y = np.max(frontier_matrix[:,0])
        min_x = np.min(frontier_matrix[:,1])
        max_x = np.max(frontier_matrix[:,1])
        frontier_height = max_y - min_y
        frontier_width  = max_x - min_x

        # padding de 1 ao redor da fronteira
        new_height = frontier_height+1
        new_width = frontier_width+1
        border_img = np.zeros((new_height, new_width))

        # reposiciona as coordernadas de frontier_matrix para o canto superior esquerdo
        frontier_matrix = frontier_matrix - [min_y, min_x]

        # transfere a fronteira para "border_img"
        for f in frontier_matrix:
            border_img[f[0], f[1]] = 1
        
        # cria máscar "mask" que será utilizada para extrair a sub-imagem da imagem original
        mask = copy.deepcopy(border_img)
        mask = ndimage.binary_fill_holes(mask).astype(int)

        # adiciona 3ª dimensão em mask3D para que o broadcast seja possível
        mask3D = np.zeros((new_height, new_width, 1))
        # pega os valores de mask
        mask3D[:,:,0] = mask
        # converte mask3D em uma matriz booleana
        mask3D = np.array(mask3D, dtype=bool)
        
        # aplica máscara "mask" sobre a imagem original, extraindo a subimagem "new_img"
        new_img = np.zeros((new_height, new_width,3))
        new_img = np.multiply(img[min_y:max_y+1, min_x:max_x+1], mask3D)
        img[min_y:max_y+1, min_x:max_x+1,:] = np.multiply(img[min_y:max_y+1, min_x:max_x+1], np.logical_not(mask3D))

        # troca fundos pretos da aplicação da máscara por fundos brancos
        new_img = np.where(mask3D==[0],[255,255,255], new_img)
        img_part = np.where(mask3D==[0], img[min_y:max_y+1, min_x:max_x+1], [255,255,255])
        img[min_y:max_y+1, min_x:max_x+1] = img_part
              
        # transforma imagem de borda em imagem RGB
        # determina os valores de "border_rgb", trocando fundo preto da imagem de borda por fundo branco e deixa o contorno preto
        border_test = np.where(border_img==0, 255, 0)
        border_rgb = np.stack((border_test, border_test, border_test),axis=-1)

        # pega o perimetro da sub-imagem
        perimeter = len(frontier)

        return (border_rgb, new_img, img, all_done, last_b_0, folha, freeman, perimeter)
    else:
        return (0,0,0, all_done, last_b_0, folha, freeman, 0)

# img_num é o inteiro correspondente à iteração do laço
def open_img_save_subimgs(img_num, df):
    str_num = str(img_num).zfill(2)
    path = "Folhas/Teste"+str_num+".png"
    img = cv2.imread(path)
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    
    all_done = False
    subimg_counter = 0
    last_b_0 = np.array([0,1])
    time_before_time = time.time()
    name = "Teste"+str_num
    while not all_done:
        t0 = time.time()
        border_rgb, new_img, img, all_done, last_b_0, folha, freeman, perimeter = segmentation(img, last_b_0)
        if not all_done and folha:
            subimg_counter+=1
            str_num_sub = str(subimg_counter).zfill(2)
            
            new_img_path = "Folhas/"+str_num+"-"+str_num_sub+".png"
            cv2.imwrite(new_img_path, new_img)
            border_path = "Folhas/"+str_num+"-"+str_num_sub+"-P.png"
            cv2.imwrite(border_path, border_rgb)
            t1 = time.time()
            
            # rotaciona a sequencia de freeman para obter o menor valor inteiro
            freeman = rot_freeman(freeman)

            # pega o vetor de diferenças da cadeia de freeman
            first_difference = first_dif(freeman)

            # adiciona a linha referente a sub-imagem ao dataframe que irá gerar o .csv
            df_row = {
                'ID Imagem': name, 
                'ID Folha': str_num_sub, 
                'Perímetro': perimeter, 
                'Cadeia de Freeman': freeman, 
                'Primeira Diferença': first_difference
            }
            df = df.append(df_row, ignore_index = True)
            '''
            print("Subimagem "str_num+"-"+str_num_sub+" salva.")
            print(f"Tempo: {t1-t0}")
            print("-"*20)
            '''
            

    time_after_time = time.time()
    print("Imagem "+str_num)
    print(f"Tempo Total: {time_after_time-time_before_time}")
    print("-"*20)
    return df

df = pd.DataFrame(columns = ['ID Imagem', 'ID Folha', 'Perímetro', 'Cadeia de Freeman', 'Primeira Diferença'])

df_data = pd.DataFrame(columns = ['ID Imagem', 'ID Folha', 'Perímetro', 'Cadeia de Freeman', 'Primeira Diferença'])
for i in range(1, 16):
    df_data = df_data.append(open_img_save_subimgs(i, df), ignore_index = True)

df_data.to_csv("Resultados.csv", index=False)

print(df_data)
