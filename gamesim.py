import random

def play(A, D, Avec, Dvec):
    """
    A: number of squadrons attacker has
    D: number of squadrons defender has
    Avec: probability distribution over placement choices for attacker s.t. Avec[i] = P((i, A-i)), for 0<=i<=A
    Dvec: probability distribution over placement choices for defender s.t. Dvec[i] = P((i, D-i)), for 0<=i<=D
    simulates game play given Avec and Dvec. 
    returns 1 if attacker wins and 0 if defender wins
    """
    if (A+1 != len(Avec) or D+1 != len(Dvec)):
        raise Exception("incompatible dimensions given for input")
    A_poss = [(i, A-i) for i in range(A+1)]
    D_poss = [(i, D-i) for i in range(D+1)]
    A_chosen = random.choices(A_poss, weights=Avec, k=1)[0] # change k for different number of selections
    D_chosen = random.choices(D_poss, weights=Dvec, k=1)[0]
    print('Attacker chose to play', A_chosen)
    print('Defender chose to play', D_chosen)
    if (A_chosen[0] > D_chosen[0] or A_chosen[1] > D_chosen[1]):
        print("Attacker wins")
        return 1 
    print("Defender wins")
    return 0

def simulate_trials(n, A, D, Avec, Dvec):
    """
    n: number of trials
    A: number of squadrons attacker has
    D: number of squadrons defender has
    Avec: probability distribution over placement choices for attacker s.t. Avec[i] = P((i, A-i)), for 0<=i<=A
    Dvec: probability distribution over placement choices for defender s.t. Dvec[i] = P((i, D-i)), for 0<=i<=D
    returns proportion of attacker wins over n trials
    """
    score = 0
    for i in range(n):
        score += play(A, D, Avec, Dvec)
    return score/n

# usage examples
Avec = [1,0,0,0,0]
Dvec = [0.5,0.5,0,0,0]
print(play(4,4,Avec,Dvec))

n=10
print('Attacker proportion of wins over', n, 'trials', simulate_trials(n,4,4,Avec,Dvec))
