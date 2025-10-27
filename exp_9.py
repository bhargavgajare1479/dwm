#exp 9

from itertools import combinations

# --- Dataset (Grocery Transactions) ---

data = [

    {"milk", "bread", "butter"},

    {"bread", "diapers", "beer", "milk"},

    {"milk", "bread", "diapers", "butter"},

    {"bread", "butter"},

    {"milk", "bread", "diapers"},

    {"bread", "beer"},

    {"milk", "diapers", "beer", "cola"},

    {"milk", "bread", "diapers", "beer"},

    {"bread", "butter", "jam"},

    {"milk", "bread", "diapers"},

]

# --- Apriori Algorithm ---

def apriori(transactions, min_sup):

    items = sorted(set().union(*transactions))

    min_count = min_sup * len(transactions)

    k, freq = 1, {}

    def get_freq(cands):

        res = {}

        for c in cands:

            count = sum(1 for t in transactions if c.issubset(t))

            if count >= min_count:

                res[frozenset(c)] = count

        return res

    cand = [frozenset([i]) for i in items]

    while cand:

        f = get_freq(cand)

        if not f: break

        freq.update(f)

        keys = list(f.keys())

        cand = [a | b for i, a in enumerate(keys) for b in keys[i+1:] if len(a | b) == k+1]

        cand = [c for c in cand if all(frozenset(s) in f for s in combinations(c, k))]

        k += 1

    return freq

# --- Main ---

if __name__ == "__main__":

    print("Grocery Store Transactions:")

    for i, t in enumerate(data, 1):

        print(f"Transaction {i}: {t}")

    min_support = float(input("\nEnter minimum support (0-1): "))

    result = apriori(data, min_support)

    print("\nFrequent Itemsets:")

    for s, count in sorted(result.items(), key=lambda x: (-len(x[0]), -x[1])):

        print(f"{set(s)} -> Support: {count/len(data):.2f}")