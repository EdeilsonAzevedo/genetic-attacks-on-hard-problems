# 🧬 Genetic Attacks on Hard Problems

Este projeto explora o uso de **Algoritmos Genéticos (AGs)** para resolver dois problemas de otimização **NP-Hard** com fortes restrições combinatórias e funcionais:

- 📊 **Seleção de Portfólio com Restrição de Cardinalidade** (baseado no modelo de Markowitz)
- 🧠 **Alocação de Tarefas com Restrições Operacionais**

Ambos os problemas são modelados como espaços de busca complexos e tratados com operadores evolutivos personalizados.


## 🚀 Visão Geral dos Problemas

### 1. **Markowitz com Cardinalidade**

Marcinho, o magnata dos imóveis, deseja investir em 5 ações dentre 20 disponíveis, buscando **maximizar o retorno esperado** e **minimizar o risco**. Cada ativo tem um retorno esperado e uma variância individual.

**Restrições:**
- Exatamente **5 ativos no portfólio**
- Cada ativo deve ter no mínimo **10% de alocação**
- As alocações devem somar exatamente **100%**
- Apenas a **variância individual** dos ativos é considerada como risco



### 2. **Alocação de Tarefas com Restrições**

Uma equipe de 10 funcionários deve executar 20 tarefas com diferentes durações. Cada funcionário tem um nível de afinidade com cada tarefa.

**Restrições:**
- Cada tarefa deve ser atribuída a **exatamente um funcionário**
- Cada funcionário deve receber entre **1 e 3 tarefas**
- A carga horária total de cada funcionário não pode ultrapassar **10 horas**

**Objetivo:** Maximizar a soma das **afinidades atribuídas**.
