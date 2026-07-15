# Rota A — plano por estágios e pontos de parada

Este plano adota como alvo editorial *Physical Review E* (IF aproximado 2) e mantém o
enquadramento estritamente mecanístico: não há alegação de vantagem quântica, de superioridade
absoluta de previsão, nem de skill absoluto em `h=64`. O caso EEG permanece limitado a segmentos
de uma única base; S é nulo no contraste causal; as formas distribuídas de kernel empatam.

## Estágio 0 — pré-requisitos e integridade

Produz:

- `results/eeg/useful_horizon_v2.csv`: maior horizonte com NRMSE médio abaixo de 1 e limite
  inferior do bootstrap pareado da melhora sobre persistência acima de 0, aplicado de forma
  idêntica a todos os modelos;
- `RESULTS.md`: narrativa derivada dos CSVs, liderada por horizonte útil e interação, com S-null,
  empate entre formas, ausência de skill longo, nível-segmento e base única explícitos;
- `scripts/verify_rotaA_gate0.py`: verificação fail-high da simetria, consistência textual,
  imutabilidade do gate empírico, testes e SHA256;
- atualização de `results/eeg/PROVENANCE.md` e `provenance/eeg_checksums.txt`.

**Para no Gate 0:** horizonte útil v2 correto/simétrico, textos consistentes, gate empírico
inalterado, `pytest` e verificador aprovados. Nenhuma derivação teórica começa antes de revisão
humana.

## Estágio 1 — teoria do kernel efetivo

Produz:

- Entradas: blob comprometido `HEAD:results/eeg/hp_selected.json`, entrada
  `single_kernel.hp`; código existente de canal, observáveis, kernels e simulador; commit Git e
  working-tree status registrados. A decisão humana fixa K=15, r=0.7 e past_mass=0.3. O arquivo
  divergente do working tree (r=0.9) não é alterado; seus resultados permanecem classificados em
  `results/eeg/_invalid_config_r09_snapshot/`.
- Scripts: `scripts/run_effective_kernel_check.py` e `scripts/run_rotaA_stage1.sh`, com
  `scripts/verify_rotaA_gate1.py` como último comando.
- `docs/effective_kernel_theory.md`: linearização CPTP, transferência correta,
  polos/estabilidade do companion, pesos geométricos, lags médios e PSD;
- `results/eeg/theory_vs_sim_check.csv`, `theory_vs_sim_responses.npz` e
  `theory_vs_sim_metadata.json`: teste confirmatório epsilon=1e-4;
- `results/eeg/theory_linearity_sweep.csv`: robustez pós-gate, sem alterar o veredito.

**Para no Gate 1:** derivação e teste teoria-vs-simulação reportados. Se a tolerância falhar, a
linearização é declarada insuficiente e o trabalho para, sem ajuste retroativo.

## Estágio 2 — bateria sintética

Produz:

- `docs/synthetic_stage2_protocol.md` e `config/rotaA_stage2_frozen.json`, congelados antes da
  simulação não linear;
- processos AR(1), AR(2)/oscilador, ruído colorido, estrutura de ordem superior e surrogate de
  fase, com splits completos disjuntos e scaling ajustado apenas no treino;
- previsões primeiro por `H_actual`, preservadas em
  `results/synth/theory_predictions_frozen.csv` com SHA256 antes da medição;
- `results/synth/theory_predictions_vs_measured.csv`, figura e `gate2_report.md`, com slopes,
  IC bootstrap, espectro companion, `T_eff`, rankings e veredito mecânico.

**Para no Gate 2:** previsão-vs-medida completa, ICs e discrepâncias documentadas. Bonn permanece
apenas como ilustração; nenhuma segunda base de EEG real é iniciada.

## Estágio 3 — recurso físico

Produz:

- adendo pós-gate que usa somente os resultados congelados do Gate 2, em
  `docs/gate2_postgate_addendum.md`, `results/synth/gate2_postgate_sensitivity.csv` e figura;
- protocolo/configuração congelados em `docs/gate3_protocol.md` e
  `config/rotaA_gate3_frozen.json`;
- tabela de memória, operações, features e medições em
  `results/resources/qrc_resource_table.csv` e `paper/tab_physical_resources.tex`;
- resultados exatos e sob shots em `results/eeg/shot_sensitivity_*.csv`, com baseline exato
  obrigatório, relatório `results/eeg/gate3_report.md` e figura PDF/PNG;
- `docs/physical_resources.md`, declarando a mistura explícita de estados passados como mecanismo
  híbrido/simulado salvo realização por ensembles, repreparação ou ancillas;
- `scripts/verify_gate3.py`, que recompõe configuração, recursos, resumos, classificações e hashes.

**Para no Gate 3:** custo, sensibilidade a shots e implementabilidade reportados, com limitações;
o comando canônico é `bash scripts/run_rotaA_stage3.sh` e para antes do manuscrito.

## Estágio 4 — manuscrito REVTeX

Produz:

- figuras APS PDF+PNG 600 dpi: kernel/polos, sintético, EEG e recursos;
- tabelas `.csv`+`.tex`: interação, horizonte útil v2, capacidade e custo;
- `paper/manuscript.tex`, teoria-primeiro, com título provisório definido no protocolo;
- Data Availability Statement e `.zenodo.json` para release imutável.

**Para no Gate 4:** manuscrito compila, DAS presente, figuras/tabelas e SHA256 completos. O
processo termina aqui; publicação ou segunda base exigem decisão separada.

## Regras transversais

Em todos os estágios: normalização causal e seleção de HP por segmento permanecem congeladas;
qualquer desvio é explícito; S-null, empate das formas, nível-segmento/base única e falta de skill
absoluto longo são sempre reportados. Cada estágio termina no próprio gate para revisão humana.
