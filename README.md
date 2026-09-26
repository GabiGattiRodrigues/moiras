# Moiras

Cases de classificação com XGBoost, todos passando pelo mesmo caminho: baseline honesto, XGBoost, calibração, ponto de corte por custo, explicação (SHAP) e monitoramento (PSI).

**Case 1 (pronto): risco de crédito** com dados reais do Kaggle, Give Me Some Credit.
Próximos: turnover (base simulada e rotulada como simulada) e recompra.

## Os cases
| Case | Base | Vencedor pela regra |
|---|---|---|
| Crédito | Give Me Some Credit (Kaggle, real) | XGBoost |
| Turnover | IBM HR Attrition (Kaggle, fictícia) | Regressão logística |
| Aquisição | Bank Marketing (UCI, real) | XGBoost |

Regra de escolha, definida antes de rodar: AUC de validação cruzada no treino; entre os modelos a menos de 1 erro-padrão do melhor, vence o mais simples. O teste só confirma.

O case de crédito ainda tem a aplicação numa fila de 101 mil propostas novas (com consulta por cliente e upload de CSV), PSI e teste de estresse.

## Rodar o app
Dê dois cliques em `iniciar.bat` (ou `pip install -r requirements.txt` e `streamlit run app.py`).

## Refazer o treino
Coloque os arquivos em `data/` (`cs-training.csv`, `cs-test.csv`, `WA_Fn-UseC_-HR-Employee-Attrition.csv`, `bank-additional-full.csv`) e rode:
`python train_credito.py --data data` (case 1), `python train_cases.py credito|turnover|aquisicao --data data`.
Os resultados vão para `artifacts/`, que é o que o app lê. O app não precisa da pasta `data/`.

## Estrutura
- `moiras/metrics.py`: KS, AUC, lift por decil, calibração, PSI e curva de lucro
- `moiras/compare.py`: a disputa entre as quatro famílias e a regra de 1 erro-padrão
- `moiras/ui_caso.py`: página genérica dos cases 2 e 3
- `train_credito.py`: limpeza, baseline logístico, XGBoost, SHAP e PSI do case de crédito
- `app.py` (navegação), `inicio.py`, `credito.py`, `turnover.py`, `aquisicao.py`: app Streamlit
- `artifacts/`: saídas do treino

Os valores de margem e prejuízo da aba Decisão são hipóteses ilustrativas. Split estratificado porque a base não tem data.

Para refazer o treino do case 1 instale também: pip install shap
