# 📊 MVP: Previsão de Customer Churn (Evasão de Clientes)

Este repositório contém o Produto Mínimo Viável (MVP) desenvolvido como requisito de avaliação para a disciplina de Sistemas de Suporte à Decisão do Departamento de Engenharia de Produção.

## 🎯 Objetivo do Projeto
O objetivo central deste projeto é desenvolver um sistema preditivo utilizando **Machine Learning Clássico** capaz de identificar antecipadamente clientes com alto risco de cancelamento de assinaturas (*Churn*). Com essa inteligência, equipes de negócios e retenção podem atuar de forma proativa, otimizando a alocação de recursos financeiros e melhorando o *Lifetime Value* (LTV) da base de clientes.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`Pipeline`, `RandomForest`, `LogisticRegression`, `RandomizedSearchCV`)
* **Visualização:** Matplotlib, Seaborn
* **Deploy:** Joblib

## 🚀 Metodologia e Pipeline de Dados
O projeto seguiu rigorosamente as etapas do ciclo de vida de ciência de dados:
1. **Extração na Nuvem:** Consumo da base de dados (Kaggle) diretamente via URL *Raw* do GitHub, garantindo reprodutibilidade.
2. **Prevenção de *Data Leakage*:** Separação estratificada entre conjuntos de Treino (70%) e Teste (30%) antes de qualquer transformação.
3. **Pré-Processamento Automatizado:** Criação de *Pipelines* com `scikit-learn` para tratamento de valores nulos (mediana e moda), padronização de escala (`StandardScaler`) e transformação de variáveis categóricas (`OneHotEncoder`).
4. **Verificação de Vazamento de dados:** Identificação e correção de um viés metodológico (*Data Leakage*) causado por variáveis irrealistas no momento da predição (`CustomerID` e `Payment Delay`).
5. **Otimização de Hiperparâmetros:** Utilização do `RandomizedSearchCV` para definir travas de complexidade no algoritmo, prevenindo o *overfitting* e forçando a generalização.

## 📈 Resultados Alcançados
O modelo final otimizado (**Random Forest Classifier**) foi submetido à base de teste e obteve as seguintes métricas:
* **Recall (Revocação) para a classe de Churn:** **96.93%**
* **Interpretação:** O sistema é capaz de identificar preventivamente cerca de 97% dos clientes que estão prestes a cancelar o serviço, permitindo ação imediata da equipe de operações.

## 📦 Deploy e Produtização
O *pipeline* completo de transformação de dados e o modelo matemático foram serializados e exportados em um único arquivo (`pipeline_churn_rf_otimizado.pkl`). Este arquivo está pronto para ser consumido e integrado à arquitetura de software ou CRMs corporativos.

## ⚙️ Como executar este projeto
1. Clone este repositório:
   ```bash
   git clone [https://github.com/danielmagliano2002/dados-mvp-churn.git](https://github.com/danielmagliano2002/dados-mvp-churn.git)
