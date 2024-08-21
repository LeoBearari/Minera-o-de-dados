import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ignorar warnings
warnings.filterwarnings('ignore')

def principal():
    # Atualize os caminhos dos arquivos de entrada e saída
    input_file = r'C:\diabetes\diabetes.csv'
    output_file_abstencao = r'C:\diabetes\diabetesClear.csv'
    output_file_normalizado = r'C:\diabetes\diabetesNormalizado.csv'  # Novo arquivo para dados normalizados

    # Defina os nomes das colunas e características
    names = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC',
             'Historico Familiar', 'Idade', 'Resultado']

    # Leitura do arquivo CSV, pulando a primeira linha de cada coluna
    df = pd.read_csv(input_file, names=names, skiprows=1, na_values='?')


    # Cópia do DataFrame original
    df_original = df.copy()

    # Convertendo colunas para tipos numéricos
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Exibição das primeiras 15 linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")

    # Exibição de informações gerais sobre os dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")

    # Descrição dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")

    # Identificação de colunas com valores faltantes
    columns_missing_value = df.columns[df.isnull().any()]
    print("COLUNAS COM VALORES FALTANTES\n")
    print(columns_missing_value)
    print("\n")

    # Escolha um método para lidar com valores ausentes (por exemplo, 'mode' para preenchimento com moda)
    method = 'median'

    for c in columns_missing_value:
        UpdateMissingValues(df, c, method)

    # Normalização Min-Max
    features_to_normalize = ['Número Gestações', 'Glucose', 'pressao Arterial', 'Expessura da Pele', 'Insulina', 'IMC',
                             'Idade']
    df_normalized = normalize_data(df, features_to_normalize)

    # Salvamento do DataFrame pré-processado em um novo arquivo CSV
    df.to_csv(output_file_abstencao, index=False)

    # Salvamento do DataFrame normalizado em um novo arquivo CSV
    df_normalized.to_csv(output_file_normalizado, index=False)


def UpdateMissingValues(df, column, method="median", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)


def normalize_data(df, features):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[features] = scaler.fit_transform(df_normalized[features])
    return df_normalized


if __name__ == "__main__":
    principal()
