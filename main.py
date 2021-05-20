import sys
from argparse import ArgumentError
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class LSA_Summarizer:
    def __init__(self):
        self.text = None
        self.sentences = None
        self.length = None

    def split_sentences(self):
        self.sentences = sent_tokenize(self.text)

    def get_word_set(self):
        return set(word_tokenize(self.text))

    def get_word_index(self, words):
        return {word: i for i, word in enumerate(words)}

    def build_matrix(self):
        self.split_sentences()
        words = self.get_word_set()
        word_index = self.get_word_index(words)
        mat = np.zeros((len(words), len(self.sentences)))
        for si, sentence in enumerate(self.sentences):
            for word in word_tokenize(sentence):
                wi = word_index[word]
                mat[wi, si] += 1
        return mat

    def steinberger_jezek(self, num_sentences):
        V = self.build_matrix()
        S = np.linalg.svd(V)[1]
        rating_vec = np.zeros(shape=(S.shape))
        for i in range(V.shape[1]):
            length = 0
            for j in range(V.shape[0]):
                length += V[j][i] ** 2
            rating_vec
            rating_vec[i] = (length * S[i] ** 2) ** (1 / 2)
        best_sentences = rating_vec.argsort()[-num_sentences:][::-1]
        best_sentences.sort()
        return best_sentences

    def murray_renals_carletta(self, num_sentences):
        V = self.build_matrix()
        S, Vt = np.linalg.svd(V)[1:]
        count_vec = np.ceil([num_sentences*(k/sum(S)) for k in S])
        selected_sent = []
        count_idx = 0
        while len(selected_sent) < num_sentences:
            if count_vec[count_idx] > 0:
                idx = np.argmax(Vt[count_idx])
                if idx not in selected_sent:
                    selected_sent.append(idx)
                    count_vec[count_idx] -= 1
                else:
                    Vt[count_idx][idx] = min(Vt[count_idx])
                pass
            else:
                count_idx += 1 
        
        selected_sent.sort()
        return selected_sent

    def summarize(self, text, length, method=1):
        self.text = text
        self.length = length
        if method == 1:
            best_sentences = self.steinberger_jezek(self.length)
        else:
            best_sentences = self.murray_renals_carletta(self.length)
        summary_sentences = [self.sentences[i] for i in best_sentences]
        summary = " ".join(summary_sentences)
        return summary


if __name__ == "__main__":

    # Leitura do texto a ser sumarizado
    if len(sys.argv) > 1:
        if sys.argv[1].endswith(".txt"):
            with open(sys.argv[1], "r", encoding="UTF-8") as f:
                text = f.read()
        else:
            raise ArgumentError("O argumento do programa deve ser um arquivo TXT")
    else:
        text = input("Digite o texto a ser sumarizado abaixo: \n\n")

    # Seleção da técnica de sumarização
    method = -1
    while method not in [1, 2]:
        print("Técnicas de sumarização disponíveis:\n")
        print("1. LSA Steinberger & Jezek")
        print("2. LSA Murray, Renals & Carletta\n")
        method = int(input("Digite o número do método a ser utilizado: "))
        if method not in [1, 2]:
            print(
                "\nInformação digitada não se refere a nenhuma das opções disponíveis. Tente novamente!\n"
            )

    print("")
    num_sentences = int(
        input("Digite o número máximo de frases que o resumo deve ter: ")
    )
    print("")

    summarizer = LSA_Summarizer()
    summary = summarizer.summarize(text, num_sentences, method)
    print(summary)
