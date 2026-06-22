# 📊 MVP: Previsão de Customer Churn com Machine Learning

> **Disciplina:** Sistemas de Suporte à Decisão — Engenharia de Produção
> **Versão:** 3.0 | **Linguagem:** Python 3.x | **Status:** ✅ Pronto para Deploy

---

## 🎯 Objetivo do Projeto

Desenvolver um sistema preditivo de **Machine Learning Clássico** capaz de identificar antecipadamente clientes com alto risco de cancelamento (*Churn*), permitindo que equipes de retenção atuem de forma **proativa** — antes que o cancelamento aconteça.

O modelo final alcança **ROC-AUC de ~0.85** e **Recall de ~99.9%** com threshold ótimo de **0.44**, detectando preventivamente cerca de 999 a cada 1000 clientes em risco de evasão.

---

## 🧠 Por que este problema importa?

| Fator | Impacto |
|---|---|
| Adquirir um cliente novo custa **5× mais** do que reter um existente | Retenção é o investimento mais eficiente |
| O Churn é uma **perda silenciosa** — o cliente não avisa | Modelos preditivos abrem uma janela de ação |
| Antecipar o cancelamento em **2–3 semanas** permite ação preventiva | Campanhas, ofertas, suporte direcionado |

---

## 🛠️ Stack Tecnológica

| Categoria | Biblioteca / Ferramenta |
|---|---|
| Linguagem | Python 3.x |
| Manipulação de dados | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` — Pipeline, RandomForest, LogisticRegression, GradientBoosting |
| Otimização | `RandomizedSearchCV` + `StratifiedKFold` |
| Testes Estatísticos | `scipy.stats` — Mann-Whitney U, Qui-Quadrado |
| Visualização | `matplotlib`, `seaborn` |
| Serialização / Deploy | `joblib` |

---

## 🏗️ Arquitetura do Pipeline

O projeto segue rigorosamente o ciclo de vida completo de ciência de dados:

```
Dados Brutos (GitHub)
       │
       ▼
 ┌─────────────────────┐
 │  Relatório de        │  Nulos · Duplicatas · Balanceamento de classes
 │  Qualidade           │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Remoção de          │  CustomerID (sem poder preditivo)
 │  Data Leakage        │  Payment Delay (consequência, não causa do Churn)
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Separação           │  70% Treino · 30% Teste · stratify=y
 │  Estratificada       │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐   Numéricas: SimpleImputer (mediana) + StandardScaler
 │  Pipeline de         │   Categóricas: SimpleImputer (moda) + OneHotEncoder
 │  Pré-Processamento   │   Garante: sem Data Leakage · reprodutível · pronto para produção
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Modelagem com       │  Regressão Logística (baseline)
 │  Validação Cruzada   │  Random Forest ← vencedor
 │  (5 Folds / 5 métri.)│  Gradient Boosting
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Otimização de       │  RandomizedSearchCV · 20 iterações · scoring=ROC-AUC
 │  Hiperparâmetros     │
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Validações          │  Learning Curve (gap=0.069 — overfitting moderado)
 │  Avançadas           │  Mann-Whitney U + Qui-Quadrado (significância estatística)
 │                      │  Análise de Threshold (ponto de corte ótimo = 0.44)
 └─────────┬───────────┘
           │
           ▼
 ┌─────────────────────┐
 │  Deploy              │  pipeline_churn_rf_v3.pkl
 │  com Metadados       │  modelo + threshold ótimo + métricas + features + versão
 └─────────────────────┘
```

---

## 📈 Resultados — Base de Teste (Hold-Out 30%)

| Métrica | Valor Real | O que significa |
|---|---|---|
| **ROC-AUC** | ~0.85 | Capacidade geral de separar as classes — 0.5 é aleatório, 1.0 é perfeito |
| **Recall** | ~99.9% | A cada 1000 churns reais, o modelo detecta 999 preventivamente |
| **Precisão** | ~69.1% | A cada 100 alertas disparados, ~69 são churns verdadeiros |
| **F1-Score** | ~0.8168 | Melhor equilíbrio entre Recall e Precisão encontrado |
| **MCC** | >0.70 | Métrica mais justa para classes desbalanceadas — confirma aprendizado real |
| **Threshold Ótimo** | 0.44 | Ponto de corte calibrado — abaixo do padrão 0.50 para maximizar Recall |

> **Nota:** Valores obtidos na execução do notebook `MVP_CHURN_V3_211043389.ipynb`.
> Execute o código para reproduzir os resultados exatos.

---

## 🔬 Validações Estatísticas (Diferenciais v3)

Este projeto vai além de métricas de ML e aplica rigor estatístico formal:

### Teste de Mann-Whitney U (variáveis numéricas)
Verifica se a distribuição de cada variável numérica é **estatisticamente diferente** entre clientes que cancelaram (Churn=1) e os que não cancelaram (Churn=0).

- **H₀:** as distribuições são iguais nos dois grupos
- **Critério:** p < 0.05 rejeita H₀ com 95% de confiança
- **Vantagem:** não assume normalidade dos dados

**Resultados obtidos:**

| Variável | p-value | Conclusão |
|---|---|---|
| Age | < 0.001 | ✅ Diferença real entre grupos |
| Tenure | < 0.001 | ✅ Diferença real entre grupos |
| Usage Frequency | < 0.001 | ✅ Diferença real entre grupos |
| Support Calls | < 0.001 | ✅ Diferença real entre grupos |
| Total Spend | < 0.001 | ✅ Diferença real entre grupos |
| Last Interaction | 0.473 | ❌ Sem diferença significativa — candidata à remoção |

### Teste Qui-Quadrado χ² (variáveis categóricas)
Verifica se existe **associação estatisticamente significativa** entre cada variável categórica e o Churn.

- **H₀:** a variável categórica é independente do Churn
- **Critério:** p < 0.05 confirma associação real

**Resultados obtidos:**

| Variável | Chi² | p-value | Conclusão |
|---|---|---|---|
| Gender | 1742.35 | < 0.001 | ✅ Associação real com o Churn |
| Subscription Type | 13.66 | 0.0011 | ✅ Associação real com o Churn |
| Contract Length | 263.11 | < 0.001 | ✅ Associação real com o Churn |

---

## 🔑 Feature Importance — O que causa o Churn

| Ranking | Variável | Importância | Importância Acumulada |
|---|---|---|---|
| 1º | Support Calls | 33.7% | 33.7% |
| 2º | Tenure | 21.1% | 54.9% |
| 3º | Usage Frequency | 11.6% | 66.5% |
| 4º | Total Spend | 8.8% | 75.3% |
| 5º | Gender_Male | 5.9% | 81.3% |
| 6º | Gender_Female | 5.6% | 86.9% |
| 7º | Age | 5.4% | 92.2% |

> **7 variáveis explicam 92% do poder preditivo do modelo.**

**Interpretação de negócio:**
- **Support Calls (33.7%):** clientes insatisfeitos ligam antes de cancelar — o suporte deficiente é preditor direto de receita perdida
- **Tenure (21.1%):** clientes novos cancelam muito mais — investir nos primeiros 90 dias de relacionamento é crítico
- **Usage Frequency (11.6%):** baixo engajamento precede o cancelamento — programas de ativação têm impacto direto na retenção

---

## 🚨 Data Leakage — Decisões de Auditoria

Duas variáveis foram identificadas e **removidas** antes de qualquer modelagem:

| Variável | Motivo da Remoção |
|---|---|
| `CustomerID` | Identificador único sem poder preditivo. O modelo poderia memorizar IDs do treino — overfitting puro. |
| `Payment Delay` | **Data Leakage**: atraso no pagamento é *consequência* do Churn (o cliente para de pagar porque já decidiu cancelar), não uma *causa*. Mantê-la seria usar informação indisponível no momento real da predição. |

---

## ⚙️ Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/danielmagliano2002/dados-mvp-churn.git
cd dados-mvp-churn
```

### 2. Instale as dependências
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### 3. Execute o notebook
Abra o arquivo `MVP_CHURN_V3_211043389.ipynb` no Jupyter Notebook ou Jupyter Lab e execute as células em ordem.

O dataset é carregado automaticamente via URL — nenhum arquivo local é necessário.

### 4. Usando o modelo exportado em produção
```python
import joblib
import pandas as pd

# Carregando o artefato completo
artefato  = joblib.load('pipeline_churn_rf_v3.pkl')
modelo    = artefato['modelo']
threshold = artefato['threshold_otimo']   # 0.44

# Fazendo previsão para novos clientes
novos_dados   = pd.DataFrame({...})  # DataFrame com as mesmas colunas do treino
probabilidade = modelo.predict_proba(novos_dados)[:, 1]
previsao      = (probabilidade >= threshold).astype(int)

# Sistema de alertas em três níveis
for i, prob in enumerate(probabilidade):
    if prob >= 0.80:
        print(f"Cliente {i}: 🔴 CRÍTICO — prob={prob:.1%} — Intervenção urgente")
    elif prob >= threshold:
        print(f"Cliente {i}: 🟡 RISCO   — prob={prob:.1%} — Campanha de retenção")
    else:
        print(f"Cliente {i}: 🟢 SEGURO  — prob={prob:.1%} — Réguas padrão")
```

---

## 📦 Estrutura do Repositório

```
dados-mvp-churn/
│
├── MVP_CHURN_V3_211043389.ipynb   # Notebook principal (v3 — versão final)
├── prova_ssd_211043389.ipynb      # Notebook original (v1)
├── pipeline_churn_rf_v3.pkl       # Modelo exportado com metadados
├── README.md                      # Este arquivo
│
└── customer_churn_dataset-
    testing-master.csv             # Dataset (também acessível via URL)
```

---

## 🔭 Próximos Passos

| Evolução | Descrição |
|---|---|
| **SHAP Values** | Explicar cada predição individual: não só "vai cancelar" mas "por quê vai cancelar" |
| **Data Drift Detection** | Monitorar quando o perfil dos clientes muda e o modelo precisa ser retreinado |
| **Calibração de Probabilidade** | `CalibratedClassifierCV` para garantir que "70% de chance" realmente signifique 70% |
| **API REST** | Expor o modelo via FastAPI para integração direta com CRM corporativo |
| **Remover Last Interaction** | p-value de 0.47 nos testes de hipótese indica que a variável não diferencia os grupos de Churn |

---

## 👤 Autor

**Daniel Magliano**
Engenharia de Produção — Sistemas de Suporte à Decisão
[github.com/danielmagliano2002](https://github.com/danielmagliano2002)

---

*MVP desenvolvido com rigor metodológico: Pipeline sem Data Leakage · Validação Cruzada Estratificada com 5 Folds · Testes de Hipótese Estatística (Mann-Whitney + Qui-Quadrado) · Diagnóstico de Overfitting com Learning Curve · Threshold Calibrado · Pronto para Produção.*
