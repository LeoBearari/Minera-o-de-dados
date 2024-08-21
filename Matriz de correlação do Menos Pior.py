import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
caminho_arquivo = r"C:\bdmineracao\diabetes_012_health_indicators_BRFSS2015.csv"
dados = pd.read_csv(caminho_arquivo)

# Selecionar as variáveis relevantes para o PCA
variaveis_selecionadas = [
    'Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

dados_selecionados = dados[variaveis_selecionadas]

# Remover linhas com valores ausentes
dados_selecionados.dropna(inplace=True)

# Calcular a matriz de correlação
matriz_correlacao = dados_selecionados.corr()

# Visualizar a matriz de correlação usando um mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=2)
dados_reduzidos = pca.fit_transform(dados_selecionados.drop(columns=['Diabetes_012']))  # Excluindo a coluna alvo

# Criar um DataFrame com os dados reduzidos
df_reduzido = pd.DataFrame(data=dados_reduzidos, columns=['PC1', 'PC2'])

# Concatenar as colunas 'Diabetes_binary' do DataFrame original com os dados reduzidos
df_final = pd.concat([dados[['Diabetes_012']], df_reduzido], axis=1)

# Exibir os dados finais com redução de dimensionalidade
print("\nDados finais com redução de dimensionalidade:")
print(df_final.head())

# Plotar o gráfico de dispersão dos dados reduzidos
plt.figure(figsize=(6, 10))
plt.scatter(df_final['PC1'], df_final['PC2'], c=df_final['Diabetes_012'], cmap='coolwarm', alpha=0.7)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Diabetes_binary em relação aos Componentes Principais')
plt.colorbar(label='Diabetes_binary')
plt.grid(True)
plt.show()
