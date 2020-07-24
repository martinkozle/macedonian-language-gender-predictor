import numpy as np

def main():
    data = np.genfromtxt('data.txt',
                         delimiter=',',
                         encoding='utf-8',
                         dtype=str)
    with open('transformed_data.csv', 'w', encoding='utf-8') as f:
        for word, g in data:
            out = '_,' * (22 - len(word)) + ','.join(list(word)) + f',{g}\n'
            f.write(out)


if __name__ == '__main__':
    main()