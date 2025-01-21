import numpy as np
import matplotlib.pyplot as plt

def histograma(dados, num_bins=50, normalizar=False):
    """
    Calcula o histograma de uma amostra de dados e retorna as frequências e os limites dos intervalos (bins).

    Parâmetros:
    ----------
    dados : array
        Dados de entrada para a criação do histograma.
    num_bins : int, opcional
        Número de intervalos (bins) para o histograma. Padrão é 50.
    normalizar : bool, opcional
        Se True, normaliza o histograma para representar uma densidade de probabilidade. Padrão é False.

    Retorna:
    -------
    frequencias : list
        Frequências de cada intervalo (bin).
    limites_bins : numpy.ndarray
        Limites dos intervalos (bins).
    """
    dados = np.array(dados)
    
    dados_ordenados = np.sort(dados)
    valor_maximo, valor_minimo = dados_ordenados.max(), dados_ordenados.min()
    largura_bin = (valor_maximo - valor_minimo) / num_bins

    frequencias = []

    # Calculando as frequências para cada intervalo (bin)
    for i in range(num_bins):
        limite_inferior = valor_minimo + i * largura_bin
        limite_superior = valor_minimo + (i + 1) * largura_bin

        # Selecionando os valores dentro do intervalo
        valores_no_bin = np.where((limite_inferior <= dados_ordenados) & (dados_ordenados < limite_superior), dados_ordenados, 0)
        frequencia = len(valores_no_bin[valores_no_bin != 0])
        frequencias.append(frequencia)

    if normalizar:
        area_total = np.sum(frequencias) * largura_bin
        frequencias = np.array(frequencias) / area_total

    limites_bins = np.linspace(valor_minimo, valor_maximo, num_bins)

    return frequencias, limites_bins



def E(x,k):
    return 0.5*k*x**2


def media(x):
    """
    Calcula a média de uma lista de números em 1D.
    
    Parâmetros:
    ----------
    x : lista ou array
        Dados de entrada para o cálculo da média.
    
    Retorna:
    -------
    média : float
        Média dos valores de x.
    """
    return sum(x)/len(x)


def desvio_padrao(x):
    """
    Calcula o desvio padrão de uma lista de números em 1D.
    
    Parâmetros:
    ----------
    x : lista ou array
        Dados de entrada para o cálculo do desvio padrão.
    
    Retorna:
    -------
    desvio_padrao : float
        Desvio padrão dos valores de x.
    """
    media_x = media(x)
    return np.sqrt(sum((x-media_x)**2)/len(x))



def V(x):
    return x**4 - 4*x**2