"""Glossário: cada conceito estatístico usado no projeto, explicado para leigos — em PT e EN."""
import streamlit as st

DASH_URL = "https://dashboardinteligente.streamlit.app"

# Cada item: (título, corpo em markdown). Corpo sempre no formato:
# o que é → analogia simples → como calcular em 1 frase → o que é bom/ruim.
CONCEITOS = {
    "pt": [
        ("🎯 AUC — a nota geral do modelo",
         "**O que é:** pegue um cliente que deu problema e um que não deu, ao acaso. AUC é a chance do modelo dar um "
         "risco maior para quem deu problema.\n\n"
         "**Para explicar a uma criança de 5 anos:** imagine dois potes de bolinhas, um de bolinhas \"boas\" e um de "
         "\"ruins\", tudo misturado. O modelo tenta adivinhar qual bolinha é de qual pote só olhando pra ela. AUC é "
         "quantas vezes, de 100 tentativas, ele acerta qual das duas é a ruim.\n\n"
         "**Escala:** 50% = está chutando cara ou coroa (não sabe nada). 100% = acerta sempre. Não existe abaixo de "
         "50% de verdade — se desse, bastaria inverter a resposta do modelo.\n\n"
         "**O que é bom:** em problemas de crédito/propensão do mundo real, 70–80% já é útil; acima de 85% é muito bom; "
         "perto de 99% costuma ser sinal de vazamento de dado (informação do futuro entrando no treino), não de um "
         "modelo perfeito."),

        ("📏 KS (Kolmogorov-Smirnov) — a maior distância entre os dois grupos",
         "**O que é:** ordene todo mundo pelo risco previsto, do mais arriscado ao mais seguro. Ande por essa lista "
         "acumulando quantos \"maus\" e quantos \"bons\" você já passou (em %). O KS é a maior diferença entre essas "
         "duas curvas em algum ponto do caminho.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como separar bolinhas boas e ruins numa fila e ver o ponto em "
         "que a pilha de ruins cresceu muito mais rápido que a de boas — essa distância é o KS.\n\n"
         "**Escala:** de 0 (as duas curvas andam juntas, o modelo não separa nada) a 100% (em algum ponto, um grupo já "
         "acabou e o outro não começou — separação perfeita).\n\n"
         "**O que é bom:** no mercado de crédito, KS acima de 30–40% já é considerado um modelo forte; acima de 60% é "
         "excelente e raro fora de dados de competição."),

        ("📈 Gini",
         "**O que é:** é a mesma informação da AUC, só que reescalada: `Gini = 2×AUC − 1`. Uma forma alternativa e "
         "comum (principalmente em crédito/bancos) de contar a mesma história.\n\n"
         "**Escala:** 0% (modelo não sabe nada, igual à AUC de 50%) a 100% (perfeito). Um AUC de 80% vira um Gini de 60%."),

        ("🎯 Brier score e log loss — o quanto a probabilidade em si está certa",
         "**O que é:** AUC e KS só olham se a **ordem** dos riscos está certa (quem é mais arriscado que quem). Brier "
         "e log loss olham se o **número da probabilidade** está certo: se o modelo disse \"10% de chance\", será que "
         "realmente acontece em 10 de cada 100 vezes?\n\n"
         "**Para explicar a uma criança de 5 anos:** é a diferença entre \"eu sei quem vai ganhar a corrida\" (ordem, "
         "AUC/KS) e \"eu sei o tempo exato que cada um vai fazer\" (o número certo, Brier/log loss).\n\n"
         "**Escala do Brier:** 0 é perfeito, quanto **menor** melhor; um modelo que sempre chuta a taxa média da base "
         "já tem um Brier de referência — bater esse número é o mínimo esperado. **Log loss** funciona parecido "
         "(menor é melhor), mas pune muito mais forte um erro **confiante** (dizer \"1% de chance\" de algo que "
         "aconteceu é caro nessa conta)."),

        ("📊 PR-AUC (curva precisão-recall)",
         "**O que é:** parecida com a AUC, mas focada só em quem o modelo aponta como risco/propenso — de quem ele "
         "aponta, quantos realmente eram (precisão), e de todos que realmente eram, quantos ele pegou (recall/revocação).\n\n"
         "**Quando importa mais que a AUC:** quando o evento é bem raro (poucos casos positivos no meio de muita gente "
         "normal) — a AUC pode parecer ótima mesmo quando, na prática, a maioria dos alarmes do modelo é falsa. A "
         "PR-AUC mostra isso melhor.\n\n"
         "**Escala:** não tem um piso fixo como os 50% da AUC — o piso é a proporção de eventos na base (quanto mais "
         "raro o evento, mais difícil manter a PR-AUC alta). Compare sempre contra esse piso, não contra 0%."),

        ("🌡️ PSI (Population Stability Index) — a base mudou desde o treino?",
         "**O que é:** compara a distribuição de uma variável (ou do score do modelo) **hoje** com a distribuição que "
         "ele viu **no treino**. Não mede se o modelo acerta; mede se o mundo que ele está vendo agora ainda parece "
         "com o mundo em que ele aprendeu.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como perceber que as roupas que você comprou ano passado não "
         "servem mais tão bem porque as pessoas da loja mudaram de tamanho médio — o modelo foi \"vestido\" pensando "
         "num público que já não é bem o de agora.\n\n"
         "**Escala (regra de bolso do mercado):** `< 0,10` estável, sem alerta · `0,10 – 0,25` mudança moderada, vale "
         "vigiar · `> 0,25` mudança grande — a população mudou o bastante para o modelo merecer revisão ou retreino."),

        ("📉 Decis e Lift — o modelo separa melhor que o acaso?",
         "**O que é:** divida todo mundo em 10 grupos (decis) do risco mais alto para o mais baixo previsto pelo "
         "modelo. **Lift** de um decil é: quantas vezes mais eventos esse grupo tem, comparado com o que teria se "
         "fosse escolhido ao acaso.\n\n"
         "**Para explicar a uma criança de 5 anos:** se você separasse 10 caixas de bombons e o modelo prometesse que "
         "a primeira caixa tem os bombons estragados, lift 3 no decil 1 quer dizer que essa caixa realmente tem 3 "
         "vezes mais bombons estragados do que uma caixa qualquer.\n\n"
         "**O que é bom:** lift bem acima de 1 no decil 1 (topo do risco) e caindo suavemente até abaixo de 1 no "
         "decil 10 (base do risco) — isso mostra que o modelo realmente concentra os casos onde diz que vai encontrá-los."),

        ("🎚️ Calibração — quando o modelo diz \"20%\", é 20% mesmo?",
         "**O que é:** agrupe os clientes pela probabilidade que o modelo deu (ex.: todos que ficaram perto de 20%) e "
         "veja qual foi a taxa **real** de evento nesse grupo. No gráfico de calibração, o ideal é uma linha reta na "
         "diagonal (probabilidade prevista = taxa real).\n\n"
         "**Por que importa além da AUC/KS:** um modelo pode ordenar muito bem quem é mais ou menos arriscado (AUC "
         "alta) e, mesmo assim, ter os números errados (dizer \"80%\" quando na verdade é 40%) — isso é péssimo "
         "quando a probabilidade vira uma decisão de negócio (preço, limite, corte), como faz este projeto."),

        ("⚖️ A regra de 1 erro-padrão (como escolhemos o modelo vencedor)",
         "**O que é:** medimos a AUC de cada modelo várias vezes (validação cruzada, embaralhando quem entra no "
         "treino) e calculamos uma média **e** uma margem de erro (o erro-padrão, EP) em cima dela — como uma pesquisa "
         "eleitoral que vem com \"margem de erro de 2 pontos\".\n\n"
         "**A regra, decidida antes de rodar qualquer modelo:** entre todos os modelos cuja nota está dentro de 1 "
         "margem de erro do melhor (ou seja, empatados na prática), escolhemos o **mais simples** "
         "(logística < árvore < random forest < XGBoost). Um modelo mais simples é mais fácil de explicar, de manter "
         "e de auditar — só vale pagar o preço de um modelo mais complexo se ele realmente entregar uma nota "
         "melhor, e não só um empate.\n\n"
         "**Por que isso importa para confiar no projeto:** essa regra foi definida **antes** de ver os resultados. "
         "É por isso que em alguns cases o XGBoost venceu e em outro a regressão logística venceu — não escolhemos o "
         "resultado, a regra escolheu."),

        ("🌳 Árvore de decisão",
         "**O que é:** uma sequência de perguntas de sim/não sobre o cliente (\"usa mais de 90% do limite?\", "
         "\"tem mais de 2 atrasos?\") até chegar numa resposta final.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como o jogo de adivinhação \"é um animal? tem 4 patas? "
         "voa?\" até descobrir a resposta.\n\n"
         "**Ponto forte:** super fácil de explicar, dá pra desenhar e mostrar pra qualquer pessoa. **Ponto fraco:** "
         "sozinha, costuma decorar demais os dados de treino e generalizar pior do que os métodos abaixo."),

        ("🌲 Random forest — um bando de árvores",
         "**O que é:** treina centenas de árvores de decisão, cada uma vendo só uma parte sorteada dos dados e das "
         "variáveis, e a resposta final é a **média** do palpite de todas.\n\n"
         "**Para explicar a uma criança de 5 anos:** em vez de perguntar para 1 pessoa, você pergunta para 500 "
         "pessoas diferentes e fica com a opinião da maioria — mesmo que cada uma erre um pouco, o erro de cada uma "
         "tende a não se repetir nas outras, então a média acerta mais que qualquer uma sozinha.\n\n"
         "**Ponto forte:** mais robusto que uma árvore sozinha, difícil de decorar o treino. **Ponto fraco:** perde "
         "a explicação simples de uma árvore — por isso mostramos as árvores individuais que mais acertam sozinhas."),

        ("🚀 XGBoost — árvores que corrigem o erro umas das outras",
         "**O que é:** em vez de treinar árvores independentes e tirar a média (como o random forest), o XGBoost "
         "treina uma árvore pequena, olha onde ela errou, treina a próxima árvore focada em corrigir esse erro, e "
         "assim por diante — a resposta final é a **soma** de todas as correções.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como fazer um desenho em várias camadas de decalque: a "
         "primeira folha já tem um rascunho, a segunda corrige onde o rascunho errou, a terceira corrige o que ainda "
         "sobrou errado, e por aí vai — no fim, as camadas empilhadas formam um desenho bem mais preciso do que "
         "qualquer folha sozinha.\n\n"
         "**Ponto forte:** costuma ser o mais preciso das 4 famílias, principalmente com muitos dados e relações "
         "complicadas entre as variáveis. **Ponto fraco:** o mais difícil de explicar sozinho — por isso mostramos "
         "SHAP (impacto de cada variável) e as árvores individuais para abrir a caixa-preta."),

        ("📐 Regressão logística",
         "**O que é:** o modelo mais simples e mais antigo dos 4. Dá um \"peso\" fixo para cada variável (ex.: "
         "\"cada atraso a mais soma X pontos de risco\") e soma tudo para chegar numa probabilidade.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como uma receita de bolo com quantidades fixas: 1 xícara de "
         "farinha sempre pesa a mesma coisa na receita, não importa o resto — aqui, cada variável sempre pesa a "
         "mesma coisa na conta, não importa o cliente.\n\n"
         "**Ponto forte:** super fácil de explicar (\"cada atraso soma tanto de risco\") e de auditar — é por isso "
         "que às vezes ela vence a disputa, mesmo sendo a mais simples: quando o problema é simples o bastante, "
         "pagar por um modelo mais complexo não compensa."),

        ("🔍 SHAP — o que pesou na decisão de UM cliente específico",
         "**O que é:** para cada cliente, o SHAP reparte a diferença entre \"o risco médio da base\" e \"o risco "
         "previsto para esse cliente\" entre as variáveis, mostrando o quanto cada uma empurrou o risco para cima ou "
         "para baixo.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como explicar por que o preço final de uma pizza deu R$ 45 em "
         "vez do preço base de R$ 30: R$ 10 a mais foi a borda recheada, R$ 8 a mais foram os 2 sabores extras, e "
         "R$ 3 a menos foi o cupom de desconto — no fim, cada ingrediente teve sua fatia explicada no preço final.\n\n"
         "**Só existe para modelos de árvore** (XGBoost, random forest, árvore) — para a regressão logística usamos "
         "o coeficiente/razão de chances, que já é diretamente interpretável."),

        ("🔍 Importância por permutação — o que pesa no modelo em geral",
         "**O que é:** diferente do SHAP (que explica 1 cliente por vez), aqui embaralhamos os valores de **uma** "
         "variável por vez (mantendo as outras intactas) e vemos o quanto a AUC do modelo piora. Se embaralhar não "
         "muda quase nada a AUC, a variável não estava sendo muito usada; se a AUC despenca, a variável é importante "
         "para o modelo inteiro, não só para 1 caso.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como tirar um ingrediente de uma receita e ver o quanto o "
         "prato piora — se tirar o sal e ninguém nota, o sal não era tão importante; se tirar e o prato fica sem "
         "graça, era essencial."),

        ("💰 Curva de lucro / decisão (margem × prejuízo)",
         "**O que é:** o modelo só entrega uma probabilidade; a decisão de negócio (aprovar ou não, abordar ou não) "
         "precisa também saber **quanto se ganha** quando acerta e **quanto se perde** quando erra. A curva de lucro "
         "testa todos os pontos de corte possíveis e mostra o resultado esperado de cada um, para achar o ponto ótimo.\n\n"
         "**Para explicar a uma criança de 5 anos:** é como decidir até que ponto vale a pena continuar apostando "
         "numa brincadeira: se ganhar vale pouco e perder custa muito, você para de apostar bem mais cedo do que se "
         "fosse o contrário — o \"corte\" é exatamente esse ponto de parar.\n\n"
         "**Por que os sliders de margem/prejuízo mudam o corte:** quanto mais caro é aprovar alguém que não paga "
         "(prejuízo alto) e quanto menor a margem de quem paga, mais **rigoroso** o corte fica (aprova menos gente, "
         "só os mais seguros); quanto maior a margem e menor o prejuízo, mais **frouxo** o corte pode ser (aprova "
         "mais gente, mesmo aceitando um pouco mais de risco)."),
    ],
    "en": [
        ("🎯 AUC — the model's overall grade",
         "**What it is:** pick one customer who defaulted and one who didn't, at random. AUC is the chance the model "
         "gives a higher risk score to the one who defaulted.\n\n"
         "**Explain it to a 5-year-old:** imagine two jars of marbles, one \"good\" and one \"bad\", all mixed "
         "together. The model tries to guess which marble came from which jar just by looking at it. AUC is how many "
         "times out of 100 tries it correctly points out the bad one.\n\n"
         "**Scale:** 50% = flipping a coin (knows nothing). 100% = always right. It can't truly go below 50% — if it "
         "did, you'd just flip the model's answer.\n\n"
         "**What's good:** in real-world credit/propensity problems, 70–80% is already useful; above 85% is very "
         "good; near 99% is usually a sign of data leakage (future information sneaking into training), not a "
         "perfect model."),

        ("📏 KS (Kolmogorov-Smirnov) — the biggest gap between the two groups",
         "**What it is:** sort everyone by predicted risk, riskiest to safest. Walk down that list accumulating what "
         "% of \"bad\" and what % of \"good\" cases you've passed so far. KS is the biggest gap between those two "
         "curves at any point along the way.\n\n"
         "**Explain it to a 5-year-old:** it's like sorting good and bad marbles into a line and finding the spot "
         "where the pile of bad ones grew much faster than the pile of good ones — that gap is the KS.\n\n"
         "**Scale:** from 0 (the two curves move together, the model separates nothing) to 100% (at some point one "
         "group is already done and the other hasn't started — perfect separation).\n\n"
         "**What's good:** in the credit industry, a KS above 30–40% is already considered a strong model; above "
         "60% is excellent and rare outside of competition data."),

        ("📈 Gini",
         "**What it is:** the same information as AUC, just rescaled: `Gini = 2×AUC − 1`. A common alternative way "
         "(mostly in credit/banking) of telling the same story.\n\n"
         "**Scale:** 0% (knows nothing, same as AUC 50%) to 100% (perfect). An AUC of 80% becomes a Gini of 60%."),

        ("🎯 Brier score and log loss — is the probability number itself right?",
         "**What it is:** AUC and KS only check whether the **ranking** of risks is right (who is riskier than "
         "whom). Brier and log loss check whether the **probability number** itself is right: if the model said "
         "\"10% chance\", does it really happen 10 times out of 100?\n\n"
         "**Explain it to a 5-year-old:** it's the difference between \"I know who's going to win the race\" "
         "(ranking, AUC/KS) and \"I know exactly how fast each one will run\" (the right number, Brier/log loss).\n\n"
         "**Brier scale:** 0 is perfect, **lower** is better; a model that always guesses the base rate already has "
         "a reference Brier score — beating that is the bare minimum. **Log loss** works similarly (lower is "
         "better), but punishes a **confident** mistake much harder (saying \"1% chance\" for something that "
         "happened is very costly under this measure)."),

        ("📊 PR-AUC (precision-recall curve)",
         "**What it is:** similar to AUC, but focused only on who the model flags as risky/likely — of those it "
         "flags, how many really were (precision), and of everyone who really was, how many it caught (recall).\n\n"
         "**When it matters more than AUC:** when the event is quite rare (few positive cases among lots of normal "
         "ones) — AUC can look great even when, in practice, most of the model's alarms are false. PR-AUC shows "
         "this better.\n\n"
         "**Scale:** it has no fixed floor like AUC's 50% — the floor is the share of events in the data (the rarer "
         "the event, the harder it is to keep PR-AUC high). Always compare against that floor, not against 0%."),

        ("🌡️ PSI (Population Stability Index) — has the population changed since training?",
         "**What it is:** compares the distribution of a variable (or the model's score) **today** with the "
         "distribution it saw **during training**. It doesn't measure whether the model is accurate; it measures "
         "whether the world it's seeing now still looks like the world it learned from.\n\n"
         "**Explain it to a 5-year-old:** it's like noticing that last year's clothes don't fit as well anymore "
         "because the store's average customer size changed — the model was \"tailored\" for an audience that isn't "
         "quite today's audience anymore.\n\n"
         "**Scale (industry rule of thumb):** `< 0.10` stable, no alert · `0.10 – 0.25` moderate shift, worth "
         "watching · `> 0.25` big shift — the population changed enough that the model deserves a review or "
         "retraining."),

        ("📉 Deciles and Lift — does the model separate better than chance?",
         "**What it is:** split everyone into 10 groups (deciles) from the highest to the lowest risk predicted by "
         "the model. A decile's **lift** is: how many times more events that group has, compared with what it would "
         "have if chosen at random.\n\n"
         "**Explain it to a 5-year-old:** if you split 10 boxes of chocolates and the model promised the first box "
         "has the spoiled ones, a lift of 3 in decile 1 means that box really does have 3 times more spoiled "
         "chocolates than a random box.\n\n"
         "**What's good:** lift well above 1 in decile 1 (top of the risk ranking), smoothly dropping below 1 by "
         "decile 10 (bottom) — that shows the model really does concentrate the cases where it says they'll be."),

        ("🎚️ Calibration — when the model says \"20%\", is it really 20%?",
         "**What it is:** group customers by the probability the model gave them (e.g., everyone near 20%) and "
         "check the **real** event rate in that group. On the calibration chart, the ideal is a straight diagonal "
         "line (predicted probability = real rate).\n\n"
         "**Why it matters beyond AUC/KS:** a model can rank risk very well (high AUC) and still have the numbers "
         "wrong (say \"80%\" when it's really 40%) — that's bad news when the probability itself becomes a business "
         "decision (price, limit, cutoff), as it does in this project."),

        ("⚖️ The 1-standard-error rule (how we pick the winning model)",
         "**What it is:** we measure each model's AUC several times (cross-validation, reshuffling who's in the "
         "training fold) and compute both an average **and** a margin of error (the standard error, SE) around it — "
         "like a poll that comes with a \"2-point margin of error\".\n\n"
         "**The rule, decided before running any model:** among all models whose score is within 1 margin of error "
         "of the best one (i.e., practically tied), we pick the **simplest** one "
         "(logistic < decision tree < random forest < XGBoost). A simpler model is easier to explain, maintain and "
         "audit — it's only worth paying for a more complex model when it truly delivers a better score, not just a "
         "tie.\n\n"
         "**Why this matters for trusting the project:** this rule was set **before** seeing the results. That's "
         "why XGBoost won in some cases and logistic regression won in another — we didn't pick the outcome, the "
         "rule did."),

        ("🌳 Decision tree",
         "**What it is:** a sequence of yes/no questions about the customer (\"uses more than 90% of the credit "
         "limit?\", \"has more than 2 late payments?\") until it reaches a final answer.\n\n"
         "**Explain it to a 5-year-old:** it's like the guessing game \"is it an animal? does it have 4 legs? can "
         "it fly?\" until you find the answer.\n\n"
         "**Strength:** super easy to explain, you can draw it and show it to anyone. **Weakness:** on its own it "
         "tends to memorize the training data too much and generalize worse than the methods below."),

        ("🌲 Random forest — a whole pack of trees",
         "**What it is:** trains hundreds of decision trees, each one seeing only a random slice of the data and "
         "variables, and the final answer is the **average** of everyone's guess.\n\n"
         "**Explain it to a 5-year-old:** instead of asking 1 person, you ask 500 different people and go with the "
         "majority opinion — even if each one is a bit wrong, their mistakes tend not to repeat across all of them, "
         "so the average beats any single person alone.\n\n"
         "**Strength:** more robust than a single tree, harder to memorize the training data. **Weakness:** loses "
         "the simple explainability of one tree — that's why we show the individual trees that scored best on their "
         "own."),

        ("🚀 XGBoost — trees that fix each other's mistakes",
         "**What it is:** instead of training independent trees and averaging them (like random forest), XGBoost "
         "trains one small tree, looks at where it was wrong, trains the next tree focused on fixing that mistake, "
         "and so on — the final answer is the **sum** of all those corrections.\n\n"
         "**Explain it to a 5-year-old:** it's like drawing a picture on several layers of tracing paper: the first "
         "sheet is already a rough sketch, the second fixes where the sketch went wrong, the third fixes what's "
         "still off, and so on — in the end, the stacked layers make a much more accurate picture than any single "
         "sheet alone.\n\n"
         "**Strength:** usually the most accurate of the 4 families, especially with lots of data and complicated "
         "relationships between variables. **Weakness:** the hardest to explain on its own — that's why we show "
         "SHAP (each variable's impact) and the individual trees to open up the black box."),

        ("📐 Logistic regression",
         "**What it is:** the simplest and oldest of the 4 models. It gives each variable a fixed \"weight\" (e.g., "
         "\"each extra late payment adds X risk points\") and adds it all up into a probability.\n\n"
         "**Explain it to a 5-year-old:** it's like a cake recipe with fixed amounts: 1 cup of flour always weighs "
         "the same in the recipe, no matter what else is in the bowl — here, each variable always counts for the "
         "same amount in the math, no matter the customer.\n\n"
         "**Strength:** super easy to explain (\"each late payment adds this much risk\") and to audit — that's "
         "why it sometimes wins the competition, even being the simplest: when the problem is simple enough, "
         "paying for a fancier model just isn't worth it."),

        ("🔍 SHAP — what drove the decision for ONE specific customer",
         "**What it is:** for each customer, SHAP splits the gap between \"the average risk in the data\" and \"the "
         "risk predicted for this customer\" across the variables, showing how much each one pushed the risk up or "
         "down.\n\n"
         "**Explain it to a 5-year-old:** it's like explaining why a pizza cost $9 instead of the base price of $6: "
         "$2 more was the stuffed crust, $1.50 more was the 2 extra toppings, and $0.50 less was the discount coupon "
         "— in the end, every ingredient got its own slice of the final price explained.\n\n"
         "**It only exists for tree-based models** (XGBoost, random forest, decision tree) — for logistic "
         "regression we use the coefficient/odds ratio instead, which is already directly interpretable."),

        ("🔍 Permutation importance — what matters to the model overall",
         "**What it is:** unlike SHAP (which explains 1 customer at a time), here we shuffle the values of **one** "
         "variable at a time (keeping the others untouched) and see how much the model's AUC gets worse. If "
         "shuffling barely changes the AUC, that variable wasn't doing much; if AUC drops a lot, the variable "
         "matters to the whole model, not just to one case.\n\n"
         "**Explain it to a 5-year-old:** it's like removing one ingredient from a recipe and seeing how much the "
         "dish suffers — if you remove the salt and nobody notices, salt wasn't that important; if the dish turns "
         "bland, it was essential."),

        ("💰 Profit curve / decision (margin × loss)",
         "**What it is:** the model only delivers a probability; the business decision (approve or not, reach out "
         "or not) also needs to know **how much you gain** when you're right and **how much you lose** when you're "
         "wrong. The profit curve tests every possible cutoff and shows the expected result of each one, to find "
         "the optimal point.\n\n"
         "**Explain it to a 5-year-old:** it's like deciding how long to keep playing a betting game: if winning "
         "pays little and losing costs a lot, you stop much earlier than if it were the other way around — the "
         "\"cutoff\" is exactly that stopping point.\n\n"
         "**Why the margin/loss sliders move the cutoff:** the more expensive it is to approve someone who won't "
         "pay (high loss) and the smaller the margin from those who do pay, the **stricter** the cutoff gets "
         "(approves fewer people, only the safest); the bigger the margin and the smaller the loss, the **looser** "
         "the cutoff can be (approves more people, even accepting a bit more risk)."),
    ],
}

TITULOS = {
    "pt": ("📚 Conceitos — o dicionário do projeto",
           "Cada número e cada gráfico usado nos 3 cases, explicado sem jargão — pensando em alguém que nunca viu "
           "esses termos antes. Volte aqui sempre que uma métrica não fizer sentido."),
    "en": ("📚 Concepts — the project's dictionary",
           "Every number and chart used across the 3 cases, explained without jargon — written for someone who's "
           "never seen these terms before. Come back here whenever a metric doesn't click."),
}

st.title("📚 Conceitos / Concepts")
st.caption("Idioma controlado pelo seletor na barra lateral. / Language is controlled by the selector in the sidebar.")
lang = st.session_state.get("lang", "pt")

titulo, sub = TITULOS[lang]
st.markdown(f"### {titulo}")
st.markdown(sub)
st.divider()

for nome, corpo in CONCEITOS[lang]:
    with st.expander(nome):
        st.markdown(corpo)

st.divider()
st.caption(("Moiras · Gabriela Gatti Rodrigues · [LinkedIn](https://linkedin.com/in/gabriela-gatti-rodrigues) · "
            f"[Dashboard Inteligente]({DASH_URL})") if lang == "pt" else
           ("Moiras · Gabriela Gatti Rodrigues · [LinkedIn](https://linkedin.com/in/gabriela-gatti-rodrigues) · "
            f"[Dashboard Inteligente (Intelligent Dashboard)]({DASH_URL})"))
