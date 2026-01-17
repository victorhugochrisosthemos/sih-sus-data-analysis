# Análise Exploratória de Dados (EDA) sobre o Sistema de Internações Hospitalares (SIH)

## Tecnologias envolvidas
- python
- pandas
- numpy
- matplotlib
- streamlit
- statsmodels
- prophet
- pysus

## Passos seguidos
- Em um Notebook com Ubuntu, usei a biblioteca pysus para baixar dados do SIH de 2019-2024, ano por ano e salvei em .csv
- Desses, selecionei somente dados de Santa Catarina
- Juntei esses dados
- Apaguei a coluna associada ao estado, validei ser dados somente de 2019-2024, dexei o CID somente com 3 <br>
dígitos para evitar granularidade e deixar mais genérico o motivo da internação
- Baixei um dicionário de CID 10
- Gerei um .csv com informações de CID e sua descrição
- Gerei outro .csv com o somatório de cada valor único de CID no dataset, com colunas de CID e total de <br>
internações, ordenados de maneira decrescente
- Depois, gerado os gráficos e as previsões

## Sobre os modelos de previsão

- São modelos estatísticos

### Holt 

(Holt’s Linear Trend Method)<br>

suavização exponencial que captura nível e tendência em séries temporais sem sazonalidade<br><br>

### Holt-Winters 

(Holt–Winters Exponential Smoothing)<br>

faz suavização exponencial com nível, tendência e sazonalidade<br><br>

### SARIMA 

(Seasonal AutoRegressive Integrated Moving Average)<br>

modelo que combina autoregressão, diferenciação, médias móveis e sazonalidade<br><br>

### Prophet 
(Prophet Forecasting Model)<br>

modelo aditivo de séries temporais com tendência, sazonalidade e tratamento de ruído<br><br>

### Para projetos futuros

Interessante considerar outras questões como o custo hospitalar e visualização desses dados, <br>
criação um grafo envolvendo cidade de moradia e local da internação a fim de verificar padrões, <br>
usar métricas de desempenho para as previsões e explorar correlações entre as colunas. Sobre e exploração <br>
de dados realizada até aqui, penso ser valido considerar a aplicação do método também em outros cenários como:<br>

- Em análise de ações na bolsa de valores, associado com outros fatores e gerando resultados com média ponderada sobre os fatores

- Acompanhar as vendas de produtos ou ocupações de hospedagem em hotéis e validar se certa ação de vendas fez diferença nos resultados

- Entender a evolução de abertura de casos de suporte sobre um rol de produtos em uma empresa
