import pandas as pd
import numpy as np
import statsmodels.api as sm

# =============================================================================
# MODELO DE REGRESSÃO LOG-LINEAR COM VARIÁVEIS DUMMY
# Sala de Redação - Rádio Gaúcha - YouTube
# Variável dependente: ln(Visualizações)
# Período: 14/06/2021 a 31/12/2025 | N = 1.148 observações
# =============================================================================

# Carregar base de dados
df = pd.read_excel('Base_Sala_YT.xlsx', sheet_name='Dados seg-sex')

# Transformação logarítmica da variável dependente
df['ln_visualizacoes'] = np.log(df['Visualizações'])
y = df['ln_visualizacoes']

# =============================================================================
# VARIÁVEIS INDEPENDENTES
# Categorias de referência omitidas:
#   - Dia da semana: sexta-feira
#   - Mês: dezembro
#   - Ano: 2025
# =============================================================================

# Eventos futebolísticos
event_dummies = [
    'Dia_jogo_gremio',
    'Pós_jogo_gremio',
    'Dia_jogo_inter',
    'Pós_jogo_inter',
    'Dia_jogo_Grenal',
    'Pós_jogo_grenal',
    'Pós_eliminação_grêmio',
    'Pós_classificação_grêmio',
    'Pós_eliminação_inter',
    'Pós_classificação_inter',
    'Pós_título_grêmio',
    'Pós_título_inter',
    'Novo_técnico_grêmio',
    'Demissão_técnico_grêmio',
    'Novo_técnico_inter',
    'Demissão_técnico_inter',
    'Dia_feriado'
]

# Dia da semana (referência: sexta-feira)
day_dummies = [
    'Programa_segunda',
    'Programa_terça',
    'Programa_quarta',
    'Programa_quinta'
]

# Ano (referência: 2025)
year_dummies = [
    'Ano_2021',
    'Ano_2022',
    'Ano_2023',
    'Ano_2024'
]

# Mês (referência: dezembro)
month_dummies = [
    'Mês_1',
    'Mês_2',
    'Mês_3',
    'Mês_4',
    'Mês_5',
    'Mês_6',
    'Mês_7',
    'Mês_8',
    'Mês_9',
    'Mês_10',
    'Mês_11'
]

# Conjunto completo de variáveis independentes
X_cols = event_dummies + day_dummies + year_dummies + month_dummies

# Adicionar intercepto (constante)
X = sm.add_constant(df[X_cols])

# =============================================================================
# ESTIMAÇÃO DO MODELO
#
# Modelo principal: OLS com erros padrão de Newey-West (HAC), maxlags=1.
# Os erros HAC, robustos a heterocedasticidade e autocorrelação, foram
# adotados de forma preventiva em razão da natureza temporal dos dados,
# na qual a autocorrelação dos resíduos é frequente, e governam toda a
# inferência sobre os coeficientes individuais (erro padrão, t, p).
# =============================================================================

modelo = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 1})

# Ajuste OLS clássico (sem HAC), usado APENAS para obter a estatística F
# clássica de significância conjunta, correspondente à fórmula
#     F = (R² / k) / [(1 - R²) / (n - k - 1)]
# apresentada no referencial teórico. Os coeficientes estimados são idênticos
# aos do modelo HAC; o que muda é a matriz de covariância e, com ela, os erros
# padrão, as estatísticas t, os p-valores e a estatística F.
modelo_classico = sm.OLS(y, X).fit()

# =============================================================================
# RESULTADOS
# =============================================================================

# Sumário completo do modelo (inferência dos coeficientes sob HAC)
print(modelo.summary())

# Tabela de coeficientes com efeito percentual (do modelo HAC)
resultados = pd.DataFrame({
    'Coeficiente': modelo.params,
    'Erro Padrão': modelo.bse,
    'Estatística t': modelo.tvalues,
    'p-valor': modelo.pvalues,
    'Efeito (%)': (np.exp(modelo.params) - 1) * 100
}).drop(index='const')

print("\n" + "="*70)
print("COEFICIENTES COM EFEITO PERCENTUAL")
print("="*70)
print(resultados.round(4).to_string())

# Indicadores de ajuste
print("\n" + "="*70)
print("INDICADORES DE AJUSTE DO MODELO")
print("="*70)
print(f"N observações                  : {int(modelo.nobs)}")
print(f"R²                             : {modelo.rsquared:.4f}")
print(f"R² ajustado                    : {modelo.rsquared_adj:.4f}")

# Estatística F de significância conjunta (clássica, OLS), correspondente
# à fórmula F = (R² / k) / [(1 - R²) / (n - k - 1)] apresentada no referencial.
# A inferência sobre os coeficientes individuais segue o modelo HAC (objeto
# 'modelo', estimado acima com cov_type='HAC').
print(f"Estatística F                  : {modelo_classico.fvalue:.2f}")
print(f"p-valor (F)                    : {modelo_classico.f_pvalue:.2e}")

print(f"Intercepto (β₀)                : {modelo.params['const']:.4f}")
print(f"Intercepto                     : {np.exp(modelo.params['const']):,.0f} visualizações (em nível)")
