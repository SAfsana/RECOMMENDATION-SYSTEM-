import time, tracemalloc, random
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

CSV = "Groceries_dataset.csv"
MIN_SUPPORT = 0.002      # lower/raise to get more/fewer rules
MIN_CONF = 0.05
TOP_N = 5
N_TEST = 500

# ---------- Load & build baskets ----------
df = pd.read_csv(CSV)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
baskets = (df.groupby(["Member_number", "Date"])["itemDescription"]
             .apply(lambda s: list(dict.fromkeys(s))))
baskets = [b for b in baskets if len(b) >= 2]
products = sorted({p for b in baskets for p in b})
print(f"{len(baskets)} transactions, {len(products)} products")

train, test = train_test_split(baskets, test_size=0.2, random_state=42)
print(f"train/test: {len(train)} / {len(test)}")

te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(train).transform(train), columns=te.columns_)

# ---------- Eclat (vertical tid-lists) ----------
def eclat(transactions, min_support):
    n = len(transactions)
    min_count = min_support * n
    tids = defaultdict(set)
    for i, t in enumerate(transactions):
        for item in t:
            tids[item].add(i)
    result = {}
    def recurse(prefix, items):
        for i, (item, tl) in enumerate(items):
            new_prefix = prefix + (item,)
            result[frozenset(new_prefix)] = len(tl) / n
            suffix = []
            for item2, tl2 in items[i + 1:]:
                inter = tl & tl2
                if len(inter) >= min_count:
                    suffix.append((item2, inter))
            if suffix:
                recurse(new_prefix, suffix)
    start = sorted([(k, v) for k, v in tids.items() if len(v) >= min_count])
    recurse((), start)
    return pd.DataFrame({"support": list(result.values()),
                         "itemsets": list(result.keys())})

# ---------- Run & time the 3 algorithms ----------
def measure(name, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    secs = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{name:10s} time {secs:6.2f} s | memory {peak/1e6:6.1f} MB | itemsets {len(out)}")
    return out

freq = {
    "Apriori":   measure("Apriori",   lambda: apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)),
    "FP-Growth": measure("FP-Growth", lambda: fpgrowth(onehot, min_support=MIN_SUPPORT, use_colnames=True)),
    "Eclat":     measure("Eclat",     lambda: eclat(train, MIN_SUPPORT)),
}

# ---------- Rules (support, confidence, lift) ----------
rules = association_rules(freq["FP-Growth"], metric="confidence", min_threshold=MIN_CONF)
print(f"{len(rules)} rules | avg support {rules.support.mean():.2%} | "
      f"avg confidence {rules.confidence.mean():.2%} | avg lift {rules.lift.mean():.2f}")
print(rules.sort_values("lift", ascending=False)
           [["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

# ---------- Recommender ----------
rule_list = [(set(r.antecedents), set(r.consequents), r.confidence * r.lift)
             for r in rules.itertuples()]

def recommend(items, n=TOP_N):
    items = set(items)
    scores = defaultdict(float)
    for ant, cons, score in rule_list:
        if ant <= items:
            for c in cons - items:
                scores[c] = max(scores[c], score)
    return [p for p, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]

print("Example:", "whole milk ->", recommend(["whole milk"]))

# ---------- Evaluation: hide last product of each test basket ----------
random.seed(42)
sample = random.sample(test, min(N_TEST, len(test)))
tp = fp = fn = tn = 0
for b in sample:
    hidden, shown = b[-1], b[:-1]
    recs = set(recommend(shown))
    for p in products:
        actual, pred = (p == hidden), (p in recs)
        tp += actual and pred
        fn += actual and not pred
        fp += (not actual) and pred
        tn += (not actual) and (not pred)

total = tp + fp + fn + tn
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
print(f"\nConfusion matrix ({total} checks)")
print(f"            Not rec.  Recommended\nNot actual  {tn:8d}  {fp:11d}\nActual      {fn:8d}  {tp:11d}")
print(f"Accuracy {(tp+tn)/total:.2%} | Precision {precision:.2%} | Recall {recall:.2%} | F1 {f1:.2%}")
