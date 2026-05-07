# ==========================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================
# pandas: Biblioteca fundamental do Python para manipulação de tabelas de dados (DataFrames).
import pandas as pd 

# TfidfVectorizer: Ferramenta do scikit-learn que transforma texto em representação numérica (vetores) calculando o peso TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer 

# cosine_similarity: Função matemática que calcula o ângulo entre dois vetores. Quanto mais próximos de 1, mais similares.
from sklearn.metrics.pairwise import cosine_similarity 

# numpy: Biblioteca para cálculos matemáticos avançados e manipulação de matrizes de forma eficiente.
import numpy as np 


# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
# pd.read_csv: Carrega os arquivos de dados para a memória. O 'sep="|"' indica que as colunas no arquivo estão separadas por uma barra vertical.
df_interno = pd.read_csv('dataset_1.csv', sep='|') 
df_digital = pd.read_csv('dataset_2.csv', sep='|')

# 'def' cria uma função personalizada chamada 'limpar_texto' que recebe a variável 'txt' (o texto bruto).
def limpar_texto(txt):
    # pd.isna verifica se o campo está vazio (nulo). Se estiver, retorna uma string vazia ("") para não quebrar o algoritmo.
    if pd.isna(txt): return ""
    # str() converte para texto, lower() deixa tudo minúsculo e strip() remove espaços vazios no começo e no fim da string.
    return str(txt).lower().strip()

# .apply() executa a nossa função 'limpar_texto' em todas as linhas da coluna 'cliente' e salva o resultado na nova coluna 'cliente_clean'.
df_interno['cliente_clean'] = df_interno['cliente'].apply(limpar_texto)
df_digital['cliente_clean'] = df_digital['cliente'].apply(limpar_texto)

# .unique() extrai apenas os nomes sem repetição. Isso economiza processamento, pois não precisamos calcular a matemática do mesmo nome duas vezes.
clientes_interno = df_interno['cliente_clean'].unique()
clientes_digital = df_digital['cliente_clean'].unique()


# ==========================================
# 2. EXPERIMENTO: TF-IDF POR PALAVRA (CLIENTES)
# ==========================================
print("--- TF-IDF POR PALAVRA ---")

# Instancia o modelo configurando o 'analyzer' para 'word', ou seja, a unidade mínima de separação será a palavra inteira.
vectorizer_word = TfidfVectorizer(analyzer='word')

# np.concatenate une as duas listas de clientes. O .fit() ensina ao modelo todas as palavras que existem no nosso "universo" de dados (cria o vocabulário).
vectorizer_word.fit(np.concatenate((clientes_interno, clientes_digital)))

# .transform() pega os textos e os converte na matriz matemática (onde cada palavra é uma coluna e cada linha é um cliente).
matriz_word_interno = vectorizer_word.transform(clientes_interno)
matriz_word_digital = vectorizer_word.transform(clientes_digital)

# Calcula o produto escalar (distância angular) entre a matriz interna e a matriz digital. Retorna uma matriz gigante de pontuações.
similaridade_word = cosine_similarity(matriz_word_interno, matriz_word_digital)

# Localiza a posição exata (índice numérico) do registro 'joao s.' na nossa lista do sistema interno.
idx_joao_s_int = list(clientes_interno).index('joao s.')
# Localiza a posição exata (índice numérico) do registro 'joao silva' na nossa lista do canal digital.
idx_joao_silva_dig = list(clientes_digital).index('joao silva')

# Acessa a matriz de resultados passando a linha de 'joao s.' e a coluna de 'joao silva' para pegar o score exato dessa comparação.
score_word = similaridade_word[idx_joao_s_int][idx_joao_silva_dig]

# Imprime o resultado na tela limitando a exibição a 4 casas decimais (.4f).
print(f"Score (Palavra): {score_word:.4f}")


# ==========================================
# 3. EXPERIMENTO: TF-IDF POR N-GRAM (CLIENTES)
# ==========================================
print("\n--- TF-IDF POR N-GRAM ---")

# Instancia o modelo, mas agora o 'analyzer' é 'char' (caracteres) e 'ngram_range=(2,3)' indica que ele vai quebrar o texto em blocos de 2 a 3 letras.
vectorizer_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2,3))

# "Ensina" o modelo com todo o vocabulário agrupado, igual fizemos no anterior, mas agora focado nos pedaços (n-grams).
vectorizer_ngram.fit(np.concatenate((clientes_interno, clientes_digital)))

# Transforma os textos dos dois sistemas nas matrizes numéricas com base nos n-grams.
matriz_ngram_interno = vectorizer_ngram.transform(clientes_interno)
matriz_ngram_digital = vectorizer_ngram.transform(clientes_digital)

# Calcula novamente a similaridade de cosseno, agora usando a nova representação matemática.
similaridade_ngram = cosine_similarity(matriz_ngram_interno, matriz_ngram_digital)

# Extrai o score utilizando os mesmos índices (posições) numéricas encontradas anteriormente.
score_ngram = similaridade_ngram[idx_joao_s_int][idx_joao_silva_dig]

print(f"Score (N-Gram): {score_ngram:.4f}")


# ==========================================
# 4. APLICAÇÃO EM MASSA (PRODUTOS / CATÁLOGO)
# ==========================================
print("\n--- BUSCA DE PRODUTOS SIMILARES ---")

# Aplica a limpeza que criamos lá em cima nas colunas de produto.
df_interno['produto_clean'] = df_interno['produto'].apply(limpar_texto)
df_digital['produto_clean'] = df_digital['produto'].apply(limpar_texto)

# Pega os produtos únicos. A regra 'if p != ""' é uma precaução extra para ignorar produtos que ficaram totalmente vazios após a limpeza.
produtos_interno = [p for p in df_interno['produto_clean'].unique() if p != ""]
produtos_digital = [p for p in df_digital['produto_clean'].unique() if p != ""]

# Cria o modelo n-gram para produtos (de 2 a 4 caracteres, pois produtos têm abreviações técnicas maiores, ex: gen10).
vectorizer_prod = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
# Aprende o vocabulário e transforma os textos em vetores na mesma linha (fit_transform otimiza o código se fôssemos usar apenas para 1 lista).
vectorizer_prod.fit(np.concatenate((produtos_interno, produtos_digital)))

# Gera as matrizes vetoriais para os produtos de ambos os sistemas.
matriz_prod_int = vectorizer_prod.transform(produtos_interno)
matriz_prod_dig = vectorizer_prod.transform(produtos_digital)

# Calcula o cosseno (similaridade) entre todos os produtos internos e digitais.
sim_prod = cosine_similarity(matriz_prod_int, matriz_prod_dig)

# Inicia um loop (laço de repetição) usando 'enumerate' para capturar o índice ('i') e o nome do produto ('prod_int') ao mesmo tempo.
for i, prod_int in enumerate(produtos_interno):
    # np.argmax olha para a linha de resultados do produto atual e pega o índice (a posição) do produto digital que teve a maior pontuação.
    best_match_idx = np.argmax(sim_prod[i]) 
    # Guarda a pontuação máxima encontrada acessando a matriz com a linha e a coluna correta.
    best_score = sim_prod[i][best_match_idx]
    # Encontra qual era o nome do produto no canal digital usando a posição identificada no argmax.
    best_match_name = produtos_digital[best_match_idx]
    
    # Se a similaridade for maior que 0.20 (20%), imprime o resultado. Isso atua como um 'threshold' para filtrar lixos e conexões fracas.
    if best_score > 0.20:
        print(f"Score: {best_score:.4f} | Interno: '{prod_int}' ===> Digital: '{best_match_name}'")