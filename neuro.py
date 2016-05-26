"""WELL WELL WELL IT'S FAMOUS HARRY POTTER"""

import numpy as np
import time

def sigma(x):  # Активационная функция. В данном случае, сигма.
    return 1 / (1 + np.exp(-x))


def deriv_sigma(x):  # Производная активационной функции. NB: аргумент — значение самой функции, не аргумента.
    return x * (1 - x)


class Perceptron:
    def __init__(self, structure):
        """

        Короче, слушай сюда.
        Сначала создаешь себе перцептрон VASYA = Perceptron([3, 2, 1])
        Это типа у него три входа, один слой на два значения и один выходной нейрон
        Потом ,берешь входные данные a = [[1, 1, 1], [2, 2, 2]] — два примера по три чиселка
        Можешь посмотреть, че он насчитает, запустив VASYA.how_about_these(a) и вывести VASYA.output на печать
        Ничего хорошего он не насчитает, сразу говорю
        Пушо его надо обучить. Херачишь массив b=[[0], [1]], и запускаешь процедурку VASYA.learh_these(a, b)
        Вот, собственно, и все.
        А, да, проверка.
        VASYA.how_close(a,b) выдаст среднеквадратичную ошибку
        VASYA.how_many_mistakes(a, b) выдаст количество неправильных ответов перцептрона

        :param structure:
         Лист вида [i, a, b, c, d... z, o] или [i, o], где
         i: количество элементов на входе перцептрона
         a..z: количество элементов в соответствующем промежуточном слое перцептрона
         o: количество элементов на выходе перцептрона (нейронов)

        """

        # input_range — число элементов на входе перцептрона в одном примере
        self.input_range = [structure[0]]
        #self.input = np.array(structure[0], ndmin=2)

        # output_range — число нейронов (выходов) у перцептрона
        self.output_range = [structure[-1]]
        self.output = np.array(structure[-1], ndmin=2)

        # middle_range — 0, если промежуточных слоев не требуется;
        #             число элементов в промежуточном слое, если он нужен один;
        #             массив вида [a, b, c], если нужно три слоя с числом элементов a, b и c, соответственно
        self.middle_range = structure[1:-1]
        self.middle = [np.array([], ndmin=2) for i in self.middle_range]

        self.values = [np.array([], ndmin=2) for i in structure]

        self.expand_range = structure # Расширенный лист размерностей, типа [i, a, b, c, o].
        self.count = len(self.expand_range)-1 # Число итераций, для предыдущего примера — 4

        # Для каждой пары нам надо будет теперь задать матрицу перехода и вектор смещения
        self.matrix = []
        self.offset = []
        for i in range(self.count): # На единицу меньше, так как матриц нужно на одну меньше
            self.matrix.append(np.array(2 * np.random.random((self.expand_range[i], self.expand_range[i+1])) - 1))
            self.offset.append(np.array(2 * np.random.random((1, self.expand_range[i+1])) - 1))

        self.alpha = 0.9  # Скорость обучения [0..1]
        self.total_error = -1

    def how_about_these(self, input):  # Считает ответ перцептрона на набор входных значений
        self.values[0] = np.array(input, ndmin=2)
        #print("values 0 : ", self.values[0].shape)
        for i in range(0, self.count):
            self.values[i+1] = sigma(np.dot(self.values[i],self.matrix[i]) + self.offset[i])
            #print("values", i+1, ": ", self.values[i+1].shape)
        self.output = self.values[-1]

    def learn_these(self, input, wanted_result, exit_by_count = True, iterations = 100, print_progress = False):  # Учит перцептрон на многих примерах
        start = time.time()
        #self.input = np.array(input)

        if exit_by_count:
            for j in range(iterations):

                self.how_about_these(input)

                self.example_range = self.values[0].shape[0]  # Число примеров
                error = [i for i in range(self.count + 1)]
                error[self.count] = np.array(deriv_sigma(self.output) * (wanted_result - self.output))

                for i in range(self.count - 1, 0, -1):
                    error[i] = deriv_sigma(self.values[i]) * np.dot(error[i + 1], self.matrix[i].T)

                for i in range(self.count):
                    self.matrix[i] += self.alpha * np.tensordot(self.values[i], error[i + 1], axes=([0], [0]))
                    self.offset[i] += self.alpha * np.sum(error[i + 1], 0)

                if print_progress:
                    if j * 10 % iterations == 0:
                        if j == 0: print("0% ", sep='', end='', flush=True)
                        else: print(" ", j * 10 // iterations, "0% ", sep='', end='', flush=True)
                    elif j * 40 % iterations == 0:
                        print(".", sep='', end='', flush=True)
            if print_progress: print(" 100%", sep='', flush=True)
            self.time = time.time() - start
            self.total_error = np.linalg.norm(wanted_result - self.output)

    def how_close (self, input, output):
        o = np.array(output)
        #self.input = np.array(input)
        self.how_about_these(input)
        return round(np.linalg.norm(o - self.output)/o.size,3)

    def how_many_mistakes (self, input, output):
        o = np.array(output)
        #self.input = np.array(input)
        self.how_about_these(input)
        return int(np.sum(np.around(np.absolute(o - self.output))))


if __name__ == '__main__':
    #np.random.seed(43)

    N1 = Perceptron([3, 3, 2])
    inputs = [[0, 0, 0],
              [0, 1, 1],
              [0, 2, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 2, 3],
              [1, 0, 1],
              [1, 1, 1],
              [1, 2, 1],
              [1, 0, 0],
              [1, 1, 2],
              [1, 2, 3],
              [1, 1, 0]]

    outputs = [[1, 0],
               [1, 0],
               [1, 1],
               [1, 0],
               [1, 0],
               [1, 1],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0],
               [0, 0],
               [0, 1],
               [0, 0]]

    testinputs = [[0, 0, 3],
                  [0, 1, 0],
                  [0, 2, 1],
                  [1, 2, 0],
                  [1, 1, 3]]

    testoutputs = [[1, 0],
                   [1, 0],
                   [1, 1],
                   [0, 1],
                   [0, 0]]


    print("Ошибка на тестовых примерах до обучения:", N1.how_close(testinputs, testoutputs))
    print("Количество неправильных ответов:", N1.how_many_mistakes(testinputs, testoutputs))

    N1.learn_these(inputs, outputs, iterations = 10000, print_progress = True)
    print("Я ебался с твоими цифрами", round(N1.time, 3), "с.")

    print("Ошибка на тестовых примерах до обучения:", N1.how_close(testinputs, testoutputs))
    print("Количество неправильных ответов:", N1.how_many_mistakes(testinputs, testoutputs))
