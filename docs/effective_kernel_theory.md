# Kernel de histórico de estados: resposta linear corrigida

## Escopo e resultado

Esta derivação descreve localmente o reservatório híbrido/simulado implementado neste
repositório. Ela não demonstra vantagem quântica, superioridade preditiva ou que uma forma
exponencial seja universalmente melhor. A repetição confirmatória carregou do blob versionado
`HEAD:results/eeg/hp_selected.json` a construção `single_kernel` com `K=15`, `r=0.7` e
`past_mass=0.3`, no commit `6b4b4ea68fd040d29729d5a8405476e14e15fd69`.

A recorrência tangente fiel reproduziu a simulação não linear dentro das quatro tolerâncias
congeladas. A fatoração externa `H_sep(z)=W_K(z)R(z)` falhou nas quatro. O veredito mecânico é
**FAIL_SEPARABLE_FACTORIZATION**: a memória distribuída atua dentro da realimentação e altera o
espectro do sistema aumentado; ela não equivale a convoluir externamente a resposta de K=0.

## Canal CPTP e linearização

Para entrada escalar fixa `u`, o canal existente é

\[
\Phi_u(\rho)=U\left[\rho_{\rm in}(u)\otimes
\operatorname{Tr}_0(\rho)\right]U^\dagger .
\]

Ele é linear, completamente positivo e preserva traço em `rho`; sua dependência em `u` é
suave e não linear. Seja `rho_*` o estado fixo em `u0=0`. Em uma base ortonormal real das 255
matrizes hermitianas sem traço, defina

\[
A=D_\rho\Phi_{u_0}|_{\rho_*},\qquad
B=D_u\Phi_u(\rho_*)|_{u_0},\qquad y_t=Cx_t,
\]

onde `C` contém as 66 expectativas Pauli. Como o canal é linear no estado, `A` é construído
aplicando o canal de entrada fixa a cada elemento da base; `B` usa diferença central. A
recorrência de primeira ordem fiel ao simulador é

\[
x_{t+1}=A\sum_{\tau=0}^{K}w_\tau x_{t-\tau}+B\,\delta u_t,
\qquad \sum_{\tau=0}^{K}w_\tau=1.
\]

As hipóteses são perturbação pequena, ponto fixo localmente atrativo e termos de ordem superior
da codificação de entrada desprezíveis na amplitude confirmatória.

## Função de transferência e teste separável

Com

\[
W_K(z)=\sum_{\tau=0}^{K}w_\tau z^{-\tau},
\]

a transformada z da recorrência, salvo um atraso monomial dependente da convenção, fornece

\[
\boxed{H_{\mathrm{actual}}(z)=C[zI-AW_K(z)]^{-1}B}.
\]

Para K=0, `R(z)=C(zI-A)^{-1}B`. Aplicar o mesmo kernel como filtro externo produziria

\[
H_{\mathrm{sep}}(z)=W_K(z)R(z),
\]

que is not generally equal a `H_actual`. Em um modo escalar `a` de `A`, os denominadores são
`z-aW_K(z)` e `z-a`, respectivamente. SymPy confirmou que a diferença genérica entre
`1/(z-aW)` e `W/(z-a)` não é zero. Igualdade exige casos especiais, como massa atrasada nula;
não é uma identidade do modelo.

## Pesos geométricos, lags e polos

Para massa atrasada `m`, `w0=1-m` e pesos atrasados proporcionais a `r^tau`,

\[
S_K(r)=\sum_{\tau=1}^{K}r^\tau=\frac{r(1-r^K)}{1-r},
\]

\[
W_K(z)=(1-m)+\frac{m}{S_K(r)}
\frac{rz^{-1}[1-(rz^{-1})^K]}{1-rz^{-1}}.
\]

Para K finito, `W_K` é FIR. O apparent pole at `z=r` da forma racional cancela com o numerador;
`r` não é um polo físico literal do sistema finito. Para cada autovalor `a` de `A`, os polos
fechados são raízes de

\[
z^{K+1}-a\sum_{\tau=0}^{K}w_\tau z^{K-\tau}=0,
\]

ou, equivalentemente, autovalores do operador companion aumentado. A condição local completa é
raio espectral do companion menor que 1. `|r|<1` garante somabilidade no limite geométrico
infinito, mas is not by itself sufficient para estabilidade da malha fechada.

No run confirmatório, o companion tem dimensão 4080, raio espectral
`0.9587240324199373` e é localmente estável. Os pesos carregados fornecem, por cálculo no código,

\[
T_{\rm eff,delayed}=\frac{\sum_{\tau=1}^{K}\tau w_\tau}
{\sum_{\tau=1}^{K}w_\tau}=3.26178020782,
\]

e lag médio incluindo o peso presente

\[
\bar\tau=\sum_{\tau=1}^{K}\tau w_\tau=0.978534062346.
\]

## PSD e limite da interpretação de forecasting

Para entrada estacionária de pequeno sinal,

\[
S_y(\omega)=H(e^{i\omega})S_u(\omega)H(e^{i\omega})^*.
\]

Logo, a memória muda polos modais e o peso espectral das features observadas. Isso motiva uma
hipótese sobre degradação com horizonte, mas não a determina: `T_eff` sozinho não fixa a
inclinação do NRMSE. Ela também depende da PSD de entrada, observabilidade, termos não lineares e
readout. A teoria é local e ainda não explica o desempenho completo no EEG. A ligação quantitativa
com forecasting deverá ser testada, se autorizada, no Estágio 2 usando `H_actual`, não a
fatoração falsificada.

## Checagem confirmatória e robustez de amplitude

O estado fixo convergiu em 296 iterações, diferença final `9.454e-14`, com tolerância `1e-13`.

| Teoria | Métrica | Erro | Tolerância | Passa |
|---|---|---:|---:|:---:|
| Tangente | impulso, Frobenius relativo | 0.000027416 | 0.01 | sim |
| Tangente | degrau, Frobenius relativo | 0.000040720 | 0.01 | sim |
| Tangente | FFT, Frobenius relativo | 0.000027416 | 0.01 | sim |
| Tangente | função de memória, L1 | 0.000009398 | 0.02 | sim |
| Separável | impulso, Frobenius relativo | 0.418614 | 0.01 | não |
| Separável | degrau, Frobenius relativo | 0.068988 | 0.01 | não |
| Separável | FFT, Frobenius relativo | 0.418614 | 0.01 | não |
| Separável | função de memória, L1 | 0.615504 | 0.02 | não |

As similaridades cosseno, somente diagnósticas, foram `0.9999999997`/`0.9999999993` para
impulso/degrau tangentes e `0.919441`/`0.998032` para o modelo separável.

No sweep secundário, todos os epsilons testados de `1e-5` a `1e-2` mantiveram ambos os erros
tangente–simulação abaixo de 0.01. Essa é apenas a faixa amostrada, não uma prova fora dela, e o
sweep não altera o veredito confirmatório.

## Classificação do resultado anterior e parada

O resultado anterior em r=0.9 permanece em
`results/eeg/_invalid_config_r09_snapshot/` como **INVALID_CONFIG**. Ele é exploratório e não é
reutilizado nas métricas acima, embora a desigualdade algébrica genérica continue correta.

O Gate 1 corrigido termina em **FAIL_SEPARABLE_FACTORIZATION**. O Estágio 2 não foi executado;
no Stage 1 stop, no Stage 2 artifacts are produced. Recursos físicos, shots e manuscrito também
não foram iniciados.
