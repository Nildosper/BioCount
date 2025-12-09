import os
from src.contagem_placa import contar_colonias

def processar_lote(pasta_imagens, diametro_mm=None):
    """
    Processa todas as imagens de uma pasta usando contar_colonias().
    Retorna uma lista com (nome_arquivo, quantidade_detectada).
    """
    resultados = []

    if not os.path.isdir(pasta_imagens):
        raise ValueError(f"Pasta não encontrada: {pasta_imagens}")

    for nome_arquivo in os.listdir(pasta_imagens):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            caminho = os.path.join(pasta_imagens, nome_arquivo)
            resultado = contar_colonias(
                caminho,
                diametro_real_mm=diametro_mm
            )
            resultados.append((nome_arquivo, resultado["quantidade"]))

    return resultados
