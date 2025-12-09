from src.contagem_placa import contar_colonias

resultado = contar_colonias("imagens/placa_teste.jpg")

print("Contagem:", resultado["quantidade"])
print("Imagem marcada:", resultado["img_saida"])
print("CSV:", resultado["csv_saida"])

