import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Carregar o conjunto de dados
caminho_arquivo = r"C:\bdmineracao\diabetes_binary_health_indicators_BRFSS2015.csv"
dados = pd.read_csv(caminho_arquivo)

# Identificação e remoção de outliers
def remove_outliers(df):
    # Calcular a mediana para "MentHlth" e "PhysHlth"
    mediana_mental = df['MentHlth'].median()
    mediana_fisica = df['PhysHlth'].median()

    # Identificar os outliers com base na mediana
    outliers = ((df['MentHlth'] < 0) | (df['MentHlth'] > mediana_mental * 3) | (df['PhysHlth'] < 0) | (df['PhysHlth'] > mediana_fisica * 3))

    # Remover as linhas que contêm outliers
    df_filtrado = df[~outliers].copy()  # Cria uma cópia do DataFrame para evitar o aviso

    return df_filtrado

# Função para normalizar os dados por escala mínima e máxima
def normalize_min_max(df):
    # Selecionar apenas as colunas numéricas para normalização
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Criar um objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizar os dados apenas nas colunas numéricas
    df.loc[:, numeric_cols] = scaler.fit_transform(df.loc[:, numeric_cols])

    return df

# Remover outliers somente em "MentHlth" e "PhysHlth"
dados_sem_outliers = remove_outliers(dados)

# Exibir as primeiras 15 linhas do conjunto de dados sem outliers
print("\nPrimeiras 15 linhas do conjunto de dados sem outliers:")
print(dados_sem_outliers.head(15))

# Aplicar a normalização por escala mínima e máxima nos dados sem outliers
dados_normalizados = normalize_min_max(dados_sem_outliers)

# Exibir as primeiras 15 linhas dos dados normalizados
print("\nPrimeiras 15 linhas dos dados normalizados:")
print(dados_normalizados.head(15))
