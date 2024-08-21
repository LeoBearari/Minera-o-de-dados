import matplotlib.pyplot as plt
import pandas as pd

# Carregar o DataFrame original
df = pd.read_csv(r'C:\diabetes\diabetesNormalizado.csv')

# Converter todas as colunas para numérico, forçando não numéricos a NaN
for column in df.columns[:-1]:  # Ignorando a última coluna 'Resultado'
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Definir as faixas etárias
bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']

# Adicionar uma coluna 'Faixa Etária' ao DataFrame
df['Faixa Etária'] = pd.cut(df['Idade'], bins=bins, labels=labels, right=False)

# Agrupar os dados por faixa etária e resultado, contando o número de pessoas em cada grupo
result_count = df.groupby(['Faixa Etária', 'Resultado']).size().unstack()

# Plotar um gráfico de barras mostrando o número de pessoas em cada faixa etária para cada resultado
result_count.plot(kind='bar', stacked=True)
plt.xlabel('Faixa Etária')
plt.ylabel('Número de Pessoas')
plt.title('Número de Pessoas por Faixa Etária e Resultado')
plt.legend(title='Resultado')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
