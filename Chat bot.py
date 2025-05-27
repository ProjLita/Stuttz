import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import nltk
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode as unidecode  # Biblioteca para remover acentos

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Base de conhecimento
book_recommendations = {
    "álgebra linear": ["Álgebra Linear - Gilbert Strang", "Introdução à Álgebra Linear - Howard Anton"],
    "banco de dados": ["Introdução a sistemas de bancos de dados - DATE, C. J", "Sistemas de banco de dados - ELMASRI, Ramez; NAVATHE, Sham.", "Sistema de banco de dados - Silberschatz, Abraham; Korth, Henry F.; Sudarshan, S."],
    "cálculo": ["Cálculo - James Stewart", "Cálculo I - Elon Lages Lima"],
    "circuitos elétricos": ["Introdução à análise de circuitos - Robert L. Boylestad", "Fundamentos de circuitos elétricos - Charles K. Alexander; Matthew N. O. Sadiku", "Teoria e problemas de circuitos elétricos - Mahmood Nahvi; Joseph Edminister", "Circuitos elétricos - James William Nilsson; Susan A. Riedel"],
    "eletromagnetismo": ["Eletromagnetismo - Joseph Edminister; Mahmood Nahvi", "Eletromagnetismo para engenheiros - Clayton R. Paul", "Elementos de eletromagnetismo - Matthew N. O. Sadiku", "Fundamentos de eletromagnetismo com aplicações em engenharia - Stuart M. Wentworth"],
    "engenharia de software": ["Princípios de análise e projeto de sistemas com UML - BEZERRA, Eduardo", "Fundamentos do desenho orientado a objeto com UML - PAGE-JONES, Meilir", "Engenharia de software: uma abordagem profissional - PRESSMAN, Roger S.; MAXIM, Bruce R."],
    "estatística": ["Estatística Básica - Wilton de Oliveira Bussab", "Probabilidade e Estatística - William W. Hines"],
    "estrutura de computadores": ["Arquitetura de sistemas operacionais - Francis Berenger Machado, Luiz Paulo Maia", "Redes de computadores e a internet: uma abordagem top-down - James F. Kurose, Keith W. Ross", "Sistemas operacionais modernos - Andrew S. Tanenbaum"],
    "estrutura de dados": ["Algoritmos - Thomas H. Cormen", "Algoritmos em linguagem C - Paulo Feofiloff", "Estruturas de dados: conceitos e técnicas de implementação - Marcos Vianna Villas", "Introdução a estruturas de dados: com técnicas de programação em C - Waldemar Celes, Renato Cerqueira, José Lucas Rangel"],
    "inteligência artificial": ["Artificial intelligence: a modern approach - Stuart J. Russell; Peter Norvig", "Inteligência artificial - Stuart J. Russell; Peter Norvig", "Inteligência artificial: ferramentas e teorias - Guilherme Bittencourt", "Fundamentos matemáticos para a ciência da computação: um tratamento moderno de matemática discreta - Judith L. Gersting", "Inteligência artificial: estruturas e estratégias para a resolução de problemas complexos - George F. Luger"],
    "práticas na engenharia": ["Gerenciamento de projetos: guia do profissional - Claudius Jordão, Marcus Possi, Volume 1", "Gerenciamento de projetos: guia do profissional - Elizabeth Borges, Marcus Possi, Volume 2", "Gerência de projetos: guia para o exame oficial do PMI - Kim Heldman"],
    "programação orientada a objetos2": ["Orientação a objetos e SOLID para ninjas - ANICHE, Maurício", "Use a cabeça: C# - STELLMAN, Andrew; GREENE, Jennifer"]
}


study_routines = {
    "álgebra linear": ["Vetores e Espaços Vetoriais", "Matrizes e Determinantes", "Sistemas Lineares", "Autovalores e Autovetores", "Transformações Lineares e revisão"],
    "banco de dados": ["Introdução a Bancos de Dados", "Arquiteturas de Banco de Dados", "Modelagem Conceitual de Dados", "Documentação de Modelos de Dados", "Modelagem Lógica de Dados", "Introdução ao SQL e ao Ambiente de banco de dados", "Data Definition Language (DDL)", "Data Manipulation Language (DML)", "Data Query Language (DQL)"],
    "cálculo": ["Limites e Continuidade", "Derivadas", "Integrais", "Séries Infinitas", "Revisão e exercícios"],
    "circuitos elétricos": ["Circuitos em Corrente Contínua - Circuitos em Série", "Circuitos em Paralelo", "Métodos de Análise", "Teoremas da Análise de Circuitos", "Circuitos RC e RL", "Circuitos Magnéticos", "Circuitos em Corrente Alternada - Correntes e Tensões Alternadas Senoidais", "Circuitos de CA Série e Paralelo em Regime Permanente", "Circuitos RLC"],
    "eletromagnetismo": ["Álgebra Vetorial", "Campos Elétricos e Magnéticos", "Eletrodinâmica", "Materiais Dielétricos e Magnéticos", "Propagação de Ondas Eletromagnéticas", "Atividades de Laboratório"],
    "engenharia de software": ["Visão Geral do processo de desenvolvimento de Software", "Ciclos de vida de Software", "Engenharia de Requisitos – Elicitação", "Engenharia de Requisitos – Especificação", "Análise Orientada a Objetos - Modelagem de classes de análise", "Análise Orientada a Objetos - Modelagem de interações", "Análise Orientada a Objetos - Modelagem de estados", "Análise Orientada a Objetos - Modelagem de atividades"],
    "estatística": ["Distribuições de Probabilidade", "Inferência Estatística", "Regressão e Correlação", "Testes de Hipóteses", "Análise de Variância e revisão"],
    "estrutura de computadores": ["História do Computador", "Principais dispositivos de hardware de um computador", "Noções de redes de computadores: protocolos, topologias e cabeamento estruturado", "Introdução aos Sistemas Operacionais", "Principais subsistemas que compõem um sistema operacional", "Máquinas virtuais", "Sistema operacional Linux e Software Livre", "Ambientes gráficos e orientados a caractere do Linux"],
    "estrutura de dados": ["Conceito de Tipos Abstratos de Dados", "Algoritmos e Estruturas de Dados", "Abstração de Dados", "Tipos de Dados e Tipos Estruturados de Dados", "Lista Contígua", "Apontadores", "Lista Encadeada", "Pilhas, Filas e Tabelas Hash", "Árvores", "Métodos de Ordenação", "Métodos de Pesquisa"],
    "inteligência artificial": ["Histórico e visão geral da área da Inteligência Artificial", "Problemas e espaço de estado", "Técnicas de busca: desinformada e heurística", "Representação e uso do conhecimento", "Regras, objetos e lógica", "Casamento de padrões", "Uso de PROLOG, LISP e Java para tratar problemas de IA", "Processamento de Linguagem Natural", "Robótica", "Redes Neurais Artificiais", "Sistemas Especialistas", "Computação Evolutiva", "Aprendizado Indutivo"],
    "práticas na engenharia": ["O que é um projeto", "Metodologia 5W2H", "Metodologia Kanban", "Ferramentas fundamentais do Excel para gerenciamento de projeto","Metodologia SWOT"],
    "programação orientada a objetos2": ["Conceitos, definições e relacionamentos da Orientação a Objetos com C#", "Coleções de dados em C#", "Trabalhando com elementos visuais", "Integrando Banco de Dados com aplicações desenvolvidas em C#", "Padrão de Projeto"],
}



# Dados de treinamento
intents = {
    "intents": [
        {"tag": "banco de dados", "patterns": ["Me indique um livro de banco de dados", "Sugira algo sobre arquitetura de dados"],
         "responses": ["Eu recomendo 'Introdução a sistemas de bancos de dados' de DATE, C. J e 'Sistemas de banco de dados' de ELMASRI, Ramez; NAVATHE, Sham."]},
        {"tag": "engenharia de software", "patterns": ["Me indique um livro de engenharia de software", "Sugira algo sobre desenvolvimento de Software"],
         "responses": ["Eu sugiro 'Princípios de análise e projeto de sistemas com UML' de BEZERRA, Eduardo e 'Fundamentos do desenho orientado a objeto com UML' de PAGE-JONES, Meilir."]},
        {"tag": "cálculo", "patterns": ["Me indique um livro de cálculo", "Sugira algo sobre cálculo"],
         "responses": ["Para cálculo, recomendo 'Cálculo' de James Stewart e 'Cálculo I' de Elon Lages Lima."]},
        {"tag": "álgebra linear", "patterns": ["Me indique um livro de álgebra linear", "Sugira algo sobre matrizes"],
         "responses": ["Eu recomendo 'Álgebra Linear' de Gilbert Strang e 'Introdução à Álgebra Linear' de Howard Anton."]},
        {"tag": "estatística", "patterns": ["Me indique um livro de estatística", "Sugira algo sobre inferência"],
         "responses": ["Sugiro 'Estatística Básica' de Wilton de Oliveira Bussab e 'Probabilidade e Estatística' de William W. Hines."]},
        {"tag": "programação orientada a objetos", "patterns": ["Me indique um livro de programação orientada a objetos", "Sugira algo sobre padrões de projeto"],
         "responses": ["Para programação orientada a objetos, recomendo 'Orientação a objetos e SOLID para ninjas' de Maurício Aniche e 'Use a cabeça: C#' de Andrew Stellman e Jennifer Greene."]},
        {"tag": "eletromagnetismo", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Eletromagnetismo?", "O que é avaliado no curso de Eletromagnetismo?"],
         "responses": ["A Unidade I aborda 'Álgebra Vetorial'. O curso avalia a compreensão dos fenômenos eletromagnéticos, as leis do Eletromagnetismo, e a relação entre circuitos eletromagnéticos e propriedades dos materiais."]},
        {"tag": "estrutura de dados", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Estrutura de Dados?", "O que é avaliado no curso de Estrutura de Dados?"],
         "responses": ["A Unidade I aborda o 'Conceito de Tipos Abstratos de Dados', incluindo algoritmos e estruturas de dados, abstração de dados, tipos de dados e tipos estruturados de dados. O curso avalia o conhecimento sobre tipos abstratos de dados e as várias estruturas de dados existentes, habilitando o aluno a utilizar esses recursos no desenvolvimento de atividades de programação."]},
        {"tag": "estrutura de computadores", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Estrutura de Computadores?", "O que é avaliado no curso de Estrutura de Computadores?"],
         "responses": ["A Unidade I aborda 'História do Computador'. O curso avalia o conhecimento sobre os principais dispositivos de hardware de um computador, noções de redes de computadores, sistemas operacionais, máquinas virtuais, e segurança em redes."]},
        {"tag": "práticas na engenharia", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Práticas na Engenharia?", "O que é avaliado no curso de Práticas na Engenharia?"],
         "responses": ["A Unidade I aborda 'O que é um projeto: A importância da gerência de um projeto da Engenharia. Conceito de gerenciamento de projeto'. O curso avalia o conhecimento sobre metodologias de gerenciamento de projetos como 5W2H, Kanban e SWOT, além de técnicas experimentais como impressora 3D, cortadora laser, soldagem, técnicas construtivas e torno CNC."]},
        {"tag": "circuitos elétricos", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Circuitos Elétricos I?", "O que é avaliado no curso de Circuitos Elétricos I?"],
         "responses": ["A Unidade I aborda 'Circuitos em Corrente Contínua - Circuitos em Série'. O curso avalia a compreensão das leis de Ohm e Kirchhoff, análise de circuitos resistivos, técnicas de medição e análise computacional de circuitos."]},
        {"tag": "inteligência artificial", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Inteligência Artificial?", "O que é avaliado no curso de Inteligência Artificial?"],
         "responses": ["A Unidade I aborda 'Histórico e visão geral da área da Inteligência Artificial'. O curso avalia a compreensão de técnicas de busca desinformada e heurística, representação e uso do conhecimento, regras, objetos e lógica, casamento de padrões, e uso de PROLOG, LISP e Java para tratar problemas de IA."]}
    ]
}


# Função de normalização para lidar com maiúsculas, minúsculas e acentos
def normalize_text(text):
    return unidecode(text.lower())  # Remove acentos e converte para minúsculas

# Processamento dos dados
words = []
classes = []
documents = []
ignore_words = ["?", "!", ",", "."]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        norm_pattern = normalize_text(pattern)  # Normaliza texto de entrada
        word_list = nltk.word_tokenize(norm_pattern)
        words.extend(word_list)
        documents.append((word_list, normalize_text(intent["tag"])))
        if normalize_text(intent["tag"]) not in classes:
            classes.append(normalize_text(intent["tag"]))

words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Criando dados de treinamento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word) for word in doc[0]]
    
    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Criando modelo de IA
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Função para processar entrada do usuário
def bag_of_words(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_category(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    return classes[np.argmax(res)]

def chatbot(): 
    print("Olá! Quais matérias você quer incluir na sua rotina de estudos?")
    selected_subjects = []
    while True:
        subject = input("Digite uma matéria (ou 'pronto' para finalizar): ")
        if subject.lower() == "pronto":
            break
        if subject in study_routines:
            selected_subjects.append(subject)

    
    study_time = int(input("Quantas horas por dia você tem para estudar? "))# PRECISO MOSTRAR A HORA NO QUADRO DE ROTINA
    topic_duration = int(input("Quantos dias você quer passar em cada tópico? (7-14 dias recomendado) "))   
    
    print("\nAqui está sua rotina de estudos personalizada:")
    days_of_week = ["\nSegunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    mixed_schedule = []
    
    topic_schedules = []
    for subject in selected_subjects:
        topics = study_routines[subject]
        for topic in topics:
            topic_schedules.extend([(subject, topic)] * topic_duration)
    
    random.shuffle(topic_schedules)  # Mistura as matérias antes de distribuí-las(PRECISO FAZER MISTURAR SÓ EM UMA SEMANA)
    
    for i, (subject, topic) in enumerate(topic_schedules):
        day = days_of_week[i % len(days_of_week)]
        print(f"{day}: {subject} - {topic}")    
    
    print("\nAqui estão os livros recomendados para suas matérias:")
    for subject in selected_subjects:
        books = book_recommendations.get(subject, ["Nenhum livro encontrado"])
        print(f"{subject}: {', '.join(books)}")

if __name__ == "__main__":
    chatbot()
