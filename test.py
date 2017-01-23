import neuro
import parse
import pandas as pd
import numpy as np

rows = 50  # Количество примеров на вход
columns = 10  # Количество чисел в примере
out_markers = [1, 2, 3, 5, 10, 30, 60]  # На каких значениях цеплять нейроны
test_rows = 10 # Количество тестовых данных

RTS = parse.Instrument('SPFB.RTS_150101_160101.csv', rows + columns + max(out_markers) + test_rows + 10)

b = np.empty((rows, columns))  # Будет инпутом
c = np.empty((rows, len(out_markers)))  # Будет аутпутом

for i in range(rows):
    # Формируем инпут: будет массив цен, поделенный на последнюю минус один.
    b[i] = RTS.close[i:i + columns] / RTS.close[i + columns - 1] - 1
    # Формируем аутпут: разница цены через эн итераций и текущей, больше или меньше нуля
    c[i] = RTS.close[i + columns + np.array(out_markers) - 1] / RTS.close[i + columns - 1] - 1

for i in range(rows):
    for j in range(len(out_markers)):
        c[i, j] = 1 if c[i, j] > 0 else 0

# То же самое для тестовых

test_b = np.empty((test_rows, columns))
test_c = np.empty((test_rows, len(out_markers)))

for i in range(rows, rows + test_rows):
    test_b[i - rows] = RTS.close[i:i + columns] / RTS.close[i + columns - 1] - 1
    # Формируем аутпут: разница цены через эн итераций и текущей, больше или меньше нуля
    test_c[i - rows] = RTS.close[i + columns + np.array(out_markers) - 1] / RTS.close[i + columns - 1] - 1

for i in range(test_rows):
    for j in range(len(out_markers)):
        test_c[i, j] = 1 if test_c[i, j] > 0 else 0


n = neuro.Perceptron([columns, columns * 2, len(out_markers)])
print("Ло тренировки средняя погрешность на ячейку", n.how_close(test_b,test_c))
print("А количество ошибок — ", n.how_many_mistakes(test_b,test_c))
n.learn_these(b, c, iterations=1000000, print_progress=True)
print("После тренировки погрешность уменьшилась до", n.how_close(test_b,test_c))
print("А число ошибок — до", n.how_many_mistakes(test_b,test_c))














# print(1)

# if __name__ == '__main__': print(2)
