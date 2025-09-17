import random
import matplotlib.pyplot as plt
import math

def generate_allocations(total, gates):
    """
    Generate all possible allocations of `total` squadrons into `gates` gates.
    Returns a list of tuples of length `gates` whose entries sum to `total`.
    """
    if gates == 1:
        return [(total,)]
    allocations = []
    for i in range(total+1):
        for rest in generate_allocations(total - i, gates - 1):
            allocations.append((i,) + rest)
    return allocations

def play(A, D, Avec, Dvec, gates=2, verbose=True):
    """
    Simulates one round of the game.
    gates: number of gates (>=2)
    Avec: probability distribution over attacker allocations (length = number of allocations for A into gates)
    Dvec: probability distribution over defender allocations (same length for D into gates)
    """
    A_poss = generate_allocations(A, gates)
    D_poss = generate_allocations(D, gates)
    print(len(A_poss))
    print(len(D_poss))
    if len(A_poss) != len(Avec) or len(D_poss) != len(Dvec):
        raise Exception("Incompatible dimensions given for input")
    
    A_chosen = random.choices(A_poss, weights=Avec, k=1)[0]
    D_chosen = random.choices(D_poss, weights=Dvec, k=1)[0]
    
    if verbose:
        print('Attacker chose to play', A_chosen)
        print('Defender chose to play', D_chosen)
    
    # attacker wins if they beat defender at ANY gate
    for i in range(gates):
        if A_chosen[i] > D_chosen[i]:
            if verbose:
                print("Attacker wins")
            return 1
    if verbose:
        print("Defender wins")
    return 0

def simulate_trials(n, A, D, Avec, Dvec, gates=2, verbose=False):
    """
    n: number of trials
    A: number of squadrons attacker has
    D: number of squadrons defender has
    Avec: probability distribution over placement choices for attacker s.t. Avec[i] = P((i, A-i)), for 0<=i<=A
    Dvec: probability distribution over placement choices for defender s.t. Dvec[i] = P((i, D-i)), for 0<=i<=D
    gates: number of gates
    returns proportion of attacker wins over n trials
    """
    score = 0
    for _ in range(n):
        score += play(A, D, Avec, Dvec, gates=gates, verbose=verbose)
    return score / n

print(len(generate_allocations(4,3)))
# usage examples
Avec = [1]+[0]*14
Dvec = [0.5,0.5]+[0]*13
print(play(4,4,Avec,Dvec, gates=3, verbose=True))

n=10
print('Attacker proportion of wins over', n, 'trials', simulate_trials(n,4,4,Avec,Dvec, gates=3, verbose=True))


# testing hypotheses with graphs

def test_strategies_two_gates(A,D):
    """
    generates graphs showing attacker success rates with different distributions against defender who uses a uniform strategy
    for simplicity, assume A aand D are even.
    """
    n = math.floor(A/(D-A+1))
    A_win_guess = n/(n+1)
    # let defender have even split strategy (assume A, D are div by 2)
    Dvec = [1/(D+1) for i in range(D+1)]
    attacker_distributions = {
        "All on gate 1": [1] + [0]*A,
        "All on gate 2": [0]*A + [1],
        "Uniform": [1/(A+1) for _ in range(A+1)],
        "Split evenly": [0]*int(A/2) + [1] + [0]*int(A/2),
        "50% all-left, 50% all-right": [0.5] + [0]*(A-1) + [0.5]
    }

    results = {}
    for name, Avec in attacker_distributions.items():
        win_prob = simulate_trials(5000, A, D, Avec, Dvec)
        results[name] = win_prob

    plt.bar(results.keys(), results.values(), color="skyblue")
    plt.ylabel("Attacker win probability")
    plt.title(f"Attacker strategies vs Uniform Defender (A={A}, D={D})")
    plt.axhline(y=A_win_guess, color="red", linestyle="--", label=f"{n}/(n+1) = {A_win_guess:.2f}")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.show()
    return

# test_strategies_two_gates(214, 298)


# Helper to build some "named" attacker strategies for arbitrary gates
def make_attacker_strategies(A: int, gates: int):
    """
    Return a dict mapping strategy name -> probability vector (length = number of allocations).
    Strategies:
      - All on gate i (for each gate)
      - Uniform over all allocations
      - Even split (if divisible)
      - Two-corner mix: 50% all on gate 0, 50% all on gate last
      - Random sample: a few random distributions for variety
    """
    allocs = generate_allocations(A, gates)
    m = len(allocs)

    strategies: Dict[str, List[float]] = {}

    # All on each single gate
    for g in range(gates):
        vec = [0.0] * m
        target = tuple(A if i == g else 0 for i in range(gates))
        idx = allocs.index(target)
        vec[idx] = 1.0
        strategies[f"All on gate {g+1}"] = vec

    # Uniform
    strategies["Uniform"] = [1.0 / m] * m

    # Even split (only if divisible)
    if A % gates == 0:
        split = tuple([A // gates] * gates)
        vec = [0.0] * m
        vec[allocs.index(split)] = 1.0
        strategies["Even split"] = vec

    # Two-corner mix (first vs last gate)
    first = tuple(A if i == 0 else 0 for i in range(gates))
    last = tuple(A if i == (gates - 1) else 0 for i in range(gates))
    vec = [0.0] * m
    vec[allocs.index(first)] = 0.5
    vec[allocs.index(last)] = 0.5
    strategies["50% first gate, 50% last gate"] = vec

    # A few random deterministic-ish distributions:
    # place A in gate 0..gates-1 but with small noise distributed to other allocations
    for r in range(min(3, gates)):
        vec = [0.0] * m
        idx = allocs.index(tuple(A if i == r else 0 for i in range(gates)))
        vec[idx] = 0.8
        # put remaining 0.2 uniformly among other allocations
        for j in range(m):
            if j != idx:
                vec[j] = 0.2 / (m - 1)
        strategies[f"Mostly gate {r+1}"] = vec

    return strategies

def test_attacker_strategies(A: int, D: int, gates: int = 2, trials: int = 5000, threshold_n: int = None, show_allocs: bool = False):
    """
    Run a set of attacker strategies vs a uniform defender distribution and plot attacker win probabilities.
    - gates: number of gates (dimension)
    - trials: simulation trials per strategy
    - threshold_n: the 'n' to use for horizontal line n/(n+1). If None, defaults to A.
    - show_allocs: if True, print the allocation lists for attacker/defender
    """
    if threshold_n is None:
        threshold_n = A
    threshold = threshold_n / (threshold_n + 1)

    # Defender uniform over allocations of D into gates
    D_allocs = generate_allocations(D, gates)
    Dvec = [1.0 / len(D_allocs)] * len(D_allocs)

    attacker_strats = make_attacker_strategies(A, gates)

    if show_allocs:
        print("Attacker allocations (all):", generate_allocations(A, gates))
        print("Defender allocations (all):", D_allocs)

    results = {}
    for name, Avec in attacker_strats.items():
        # quick sanity check: normalize if not summing exactly to 1 due to float error
        total = sum(Avec)
        if total <= 0:
            raise ValueError(f"Strategy {name} has nonpositive total weight")
        Avec_norm = [x / total for x in Avec]

        win_prob = simulate_trials(trials, A, D, Avec_norm, Dvec, gates=gates, verbose=False)
        results[name] = win_prob
        print(f"{name}: win_prob = {win_prob:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color="skyblue")
    plt.ylabel("Attacker win probability")
    plt.title(f"Attacker strategies vs Uniform Defender (A={A}, D={D}, gates={gates})")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

# Example calls
# 2-gate example (backwards compatible)
# print("=== 2 gates example ===")
# res = test_strategies(A=10, D=10, gates=2, trials=2000)

# 3-gate example
# print("\n=== 3 gates example ===")
# res3 = test_attacker_strategies(A=15, D=35, gates=3, trials=2000, show_allocs=True)


def make_defender_strategies(D: int, gates: int):
    """
    Build a map of defender strategy-name -> weight vector (length = number of allocations).
    Ensures no all-zero weight vector is returned.
    """
    allocs = generate_allocations(D, gates)
    m = len(allocs)
    strategies = {}

    # All-in on each gate
    for g in range(gates):
        vec = [0.0] * m
        target = tuple(D if i == g else 0 for i in range(gates))
        idx = allocs.index(target)
        vec[idx] = 1.0
        strategies[f"All on gate {g+1}"] = vec

    # Uniform
    strategies["Uniform"] = [1.0 / m] * m

    # Exact even split only if divisible
    if D % gates == 0:
        split = tuple([D // gates] * gates)
        vec = [0.0] * m
        vec[allocs.index(split)] = 1.0
        strategies["Even split"] = vec
    else:
        # make an approx-even allocation by distributing remainder across the first r gates
        base = D // gates
        r = D % gates
        approx = tuple([base + 1 if i < r else base for i in range(gates)])
        vec = [0.0] * m
        vec[allocs.index(approx)] = 1.0
        strategies[f"Approx even (+{r})"] = vec

    # 50% first gate, 50% last gate
    first = tuple(D if i == 0 else 0 for i in range(gates))
    last  = tuple(D if i == (gates - 1) else 0 for i in range(gates))
    vec = [0.0] * m
    vec[allocs.index(first)] = 0.5
    vec[allocs.index(last)] = 0.5
    strategies["50% first, 50% last"] = vec

    # A few "mostly one gate" mixes (like attacker)
    for r in range(min(3, gates)):
        vec = [0.0] * m
        idx = allocs.index(tuple(D if i == r else 0 for i in range(gates)))
        vec[idx] = 0.8
        for j in range(m):
            if j != idx:
                vec[j] = 0.2 / (m - 1)
        strategies[f"Mostly gate {r+1}"] = vec

    return strategies


def test_defender_strategies(A: int, D: int, gates: int = 2, trials: int = 5000, show_allocs: bool = False):
    """
    Tests defender strategies against a uniform attacker (attacker uniform across allocations).
    Normalizes strategy vectors and skips any zero-weight strategy with a visible warning.
    """
    A_poss = generate_allocations(A, gates)
    D_poss = generate_allocations(D, gates)

    Avec = [1.0 / len(A_poss)] * len(A_poss)  # uniform attacker

    defender_strats = make_defender_strategies(D, gates)

    if show_allocs:
        print("Attacker allocations:", A_poss)
        print("Defender allocations:", D_poss)

    results = {}
    for name, Dvec in defender_strats.items():
        total = sum(Dvec)
        if total <= 0:
            print(f"Skipping strategy '{name}' because weights sum to 0 (would crash).")
            continue
        Dvec_norm = [w / total for w in Dvec]
        win_prob = simulate_trials(trials, A, D, Avec, Dvec_norm, gates=gates, verbose=False)
        results[name] = win_prob
        print(f"{name}: win_prob = {win_prob:.4f}")

    # Plot results
    plt.figure(figsize=(9, 4.5))
    plt.bar(results.keys(), results.values(), color="lightcoral")
    plt.ylabel("Attacker win probability")
    plt.title(f"Defender strategies vs Uniform Attacker (A={A}, D={D}, gates={gates})")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return results


# Example usage
test_defender_strategies(15, 35, gates=3)

