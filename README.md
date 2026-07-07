Treinar uma rede neural que preveja o investimento utilizando o dataset do banco

--------------- Transcriçao do audio ------------------------------

📘 Trabalho Final – Redes Neurais e Dataset Bancário
🎯 Objetivo
Treinar uma rede neural capaz de prever se um cliente de banco irá realizar um depósito a prazo (term deposit), com base em seu perfil e histórico de contatos. O foco é maximizar a precisão e, principalmente, aumentar o número de verdadeiros positivos (clientes previstos como investidores que realmente investem).

📂 Dataset
Arquivos disponíveis:

bank.csv → contém 10% dos exemplos (4.521 registros).

bank-full.csv → contém todos os exemplos (45.000+ registros).

Uso:

Treinar a rede com bank.csv.

Validar/testar com bank-full.csv.

🧩 Variáveis de entrada
Cada linha representa um cliente, com atributos como:

Idade

Job (tipo de emprego)

Marital (estado civil)

Education (nível de educação)

Default (se possui crédito em default)

Balance (saldo em conta)

Housing (se possui casa)

Loan (se possui empréstimo)

Contact (tipo de contato anterior)

Day/Month (último contato)

Duration (duração da ligação anterior)

Campaign (campanha que originou o contato)

Pdays (dias desde último contato)

Previous (número de contatos anteriores)

Outcome (resultado dos contatos anteriores)

Target Sim → se o cliente fez ou não o depósito.

🔧 Pré-processamento
Atributos numéricos: normalizar para valores entre 0 e 1.

Atributos categóricos:

Binários (ex.: casado/sim ou não) → codificação 0/1.

Multiclasse sem ordem (ex.: tipo de emprego) → usar One-Hot Encoding (um neurônio por classe).

Variáveis ordinais/contínuas (ex.: meses) → podem ser representadas em escala numérica ou também via One-Hot, dependendo do teste.

🧠 Rede Neural
Entrada: todos os atributos processados.

Saída: prever se o cliente fará o depósito (yes ou no).

Configuração da saída:

Pode ser um neurônio binário (0 = não, 1 = sim).

Ou dois neurônios com Softmax (um para cada classe).

Testar qual abordagem gera melhor resultado.

📊 Avaliação
Treinar com bank.csv (subset).

Validar/testar com bank-full.csv (dataset completo).

Métricas principais:

Precisão geral.

Taxa de verdadeiros positivos (clientes previstos como investidores que realmente investiram).

📅 Entrega
Trabalho final, com peso significativo na nota.

Prazo: até a próxima semana, no final da aula.

Durante a aula, haverá suporte para dúvidas e ajustes.
