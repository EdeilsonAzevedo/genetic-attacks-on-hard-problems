import pandas as pd
import numpy as np
import random
# Definir funcionários e tarefas
funcionarios = ["Ana", "Bruno", "Carla", "Diego", "Eduardo", "Fernanda", "Gustavo", "Helena", "Igor", "Júlia"]
tarefas = [f"T{i+1}" for i in range(20)]

# Gerar afinidades (valores de 0 a 10)
np.random.seed(44)  # para reprodutibilidade
afinidades = pd.DataFrame(np.random.randint(0, 11, size=(len(funcionarios), len(tarefas))),
                          index=funcionarios, columns=tarefas)

# Gerar durações de tarefas (valores de 1 a 8 horas)
duracoes = pd.DataFrame({
    "Tarefa": tarefas,
    "Duracao_horas": random.choices(range(1, 9), k=len(tarefas),
                                      weights=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])
})

# Salvar arquivos
afinidades_path = "./data/afinidades.csv"
duracoes_path = "./data/duracoes.csv"



if __name__ == "__main__":
    afinidades.to_csv(afinidades_path)
    duracoes.to_csv(duracoes_path, index=False)