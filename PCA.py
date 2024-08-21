import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Função principal
def main():
    # Caminho para o arquivo de entrada
    input_file = r'C:\diabetes\diabetesNormalizado.csv'




    # Lendo o arquivo CSV para um DataFrame
    df = pd.read_csv(input_file)
    # Converter todas as colunas para numérico, forçando não numéricos a NaN
    for column in df.columns[:-1]:  # Ignorando a última coluna 'Resultado'
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Exibindo informações sobre o DataFrame original
    show_dataframe_info(df, "DataFrame Original")

    # Selecionando características e alvo
    features = df.select_dtypes(include=['float64', 'int64']).columns
    target = 'Resultado'
    x = df[features].values
    y = df[target].values

    # Manipulando valores ausentes
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)

    # Padronizando as características
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    # PCA 2D
    visualize_pca_2d(x_scaled, y)


# Função para exibir informações sobre um DataFrame
def show_dataframe_info(df, message=""):
    print(message + "\n")
    print(df.info())  # Informações gerais sobre o DataFrame
    print(df.describe())  # Estatísticas descritivas do DataFrame
    print(df.head(10))  # Exibindo as primeiras linhas do DataFrame
    print("\n")

    # Exibição dos valores mínimo e máximo de cada característica
    print("Valores Mínimos das Características:")
    print(df.min())
    print("\nValores Máximos das Características:")
    print(df.max())
    print("\n")


# Função para visualizar o PCA em 2D
def visualize_pca_2d(x, y):
    # PCA 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)

    # Plotando os resultados do PCA 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap=plt.cm.get_cmap('viridis', 2), alpha=0.5)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('PCA 2D')
    plt.colorbar(ticks=[0, 1], label='Resultado')
    plt.grid()
    plt.show()


# Chamada da função principal
if __name__ == "__main__":
    main()
