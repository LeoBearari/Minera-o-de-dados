import pandas as pd

# Carregar o conjunto de dados
caminho_arquivo = r"C:\bdmineracao\diabetes_binary_health_indicators_BRFSS2015.csv"
dados = pd.read_csv(caminho_arquivo)

# Exibir os nomes das colunas
print("Nomes das colunas:")
print(dados.columns)

# Identificação e remoção de outliers
def remove_outliers(df):
    # Calcular a mediana para "MentHlth" e "PhysHlth"
    mediana_mental = df['MentHlth'].median()
    mediana_fisica = df['PhysHlth'].median()

    # Identificar os outliers com base na mediana
    outliers = ((df['MentHlth'] < 0) | (df['MentHlth'] > mediana_mental * 3) | (df['PhysHlth'] < 0) | (df['PhysHlth'] > mediana_fisica * 3))

    # Remover as linhas que contêm outliers
    df_filtrado = df[~outliers]

    return df_filtrado

# Remover outliers somente em "MentHlth" e "PhysHlth"
dados_sem_outliers = remove_outliers(dados)

# Exibir as primeiras 15 linhas do conjunto de dados sem outliers
print("\nPrimeiras 15 linhas do conjunto de dados sem outliers:")
print(dados_sem_outliers.head(25))
