def print_table(A, D):
    for a in range(A+1):
        for d in range(D+1):
            attacked = a > d or (A - a) > (D - d)
            print(1 if attacked else 0, end=' ')
        print()

#A, D = list(map(int, input('A D = ').split()))
#print('Attacked?')
#print_table(A, D)

print('Attack succeeded?')
N = 5
for A in range(N+1):
    D = N
    print('=' * 10)
    print('A D =', A, D)
    print('-' * 10)
    print_table(A, D)
