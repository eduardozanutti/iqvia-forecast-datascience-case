# IQVIA — Weekly Demand Forecasting

Solução para o case de previsão semanal de demanda de produtos farmacêuticos por distribuidor e região geográfica.

## Objetivo

Construir um modelo preditivo para prever a quantidade de unidades vendidas (`units_qty`) por semana, segmentado por distribuidor (`dsupp_id`) e região (`region_nm`).

## Estrutura do Projeto

```
├── config.yaml                  # Parâmetros globais do projeto
├── requirements.txt             # Dependências Python
├── docs/                        # Documentação do case
├── data/
│   ├── raw/                     # Dados originais (não modificar)
│   ├── bronze/                  # Ingestão bruta
│   ├── silver/                  # Dados limpos e transformados
│   └── gold/                    # Dados prontos para modelagem
├── notebooks/
│   ├── data_engineering/        # Pipeline de dados (bronze → silver → gold)
│   └── data_science/            # EDA e modelagem
├── src/
│   ├── preparation/             # Funções de preparação de dados
│   ├── train/                   # Treinamento de modelos
│   └── predict/                 # Predição e avaliação
├── models/                      # Modelos treinados serializados
├── experiments/                 # Rastreamento de experimentos
└── presentation/                # Apresentação final (.ppt)
```

## Como Reproduzir

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Coloque os dados brutos em `data/raw/`.

3. Execute os notebooks de engenharia de dados em ordem:
   - `notebooks/data_engineering/1-bronze_layer.ipynb`
   - `notebooks/data_engineering/2-silver_layer.ipynb`
   - `notebooks/data_engineering/3-gold_layer.ipynb`

4. Execute os notebooks de ciência de dados:
   - `notebooks/data_science/eda.ipynb`
   - `notebooks/data_science/model_building.ipynb`

## Tarefas

- [x] Estrutura do projeto
- [x] Preparação dos dados
- [x] Análise exploratória
- [ ] Construção e avaliação do modelo
- [ ] Apresentação de resultados
