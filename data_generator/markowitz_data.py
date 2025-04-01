import numpy as np
import pandas as pd

# Reprodutibilidade
np.random.seed(42)

# Número de ativos
n_assets = 20

# Retornos esperados realistas (5% a 20%)
expected_returns = np.round(np.random.uniform(0.05, 0.20, size=n_assets), 4)

# Volatilidades (desvio padrão anual) entre 10% e 40%
volatilities = np.random.uniform(0.10, 0.40, size=n_assets)

# Variâncias anuais individuais (sigma^2)
variances = np.round(volatilities**2, 6)

# Construir DataFrame
df_ativos = pd.DataFrame({
    "Ativo": [f"A{i+1}" for i in range(n_assets)],
    "Retorno_Esperado_Anual": expected_returns,
    "Variancia_Anual": variances
})

# Salvar CSV
csv_path = "./data/ativos_simplificado.csv"

if __name__ == "__main__":
    df_ativos.to_csv(csv_path, index=False)

