import json
import random
import math
import json
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori


def unpack_dual_itemset(itemset, bops=["accepted", "rejected"]):
    iset = tuple(itemset)
    target = None
    for op in bops:
        if op in itemset:
            target = op
            break
    if target is None:
        return iset
    a = iset[0]
    b = iset[1]
    if a is target:
        return b, target
    return a, b


def contingency_table(bdf, itemset):
    N = len(bdf)
    a, b = unpack_dual_itemset(itemset)
    f11 = len(bdf.query("{} == 1.0 and {} == 1.0".format(a, b)))
    f10 = len(bdf.query("{} == 1.0 and {} == 0.0".format(a, b)))
    f01 = len(bdf.query("{} == 0.0 and {} == 1.0".format(a, b)))
    f00 = len(bdf.query("{} == 0.0 and {} == 0.0".format(a, b)))
    f1p = f11 + f10
    f0p = f01 + f00
    fp1 = f11 + f01
    fp0 = f10 + f00
    return f11, f10, f01, f00


def show_contingency_table(itemset, f11, f10, f01, f00):
    N = len(bdf)
    a, b = unpack_dual_itemset(itemset)
    f1p = f11 + f10
    f0p = f01 + f00
    fp1 = f11 + f01
    fp0 = f10 + f00
    print("A = {}".format(a))
    print("B = {}".format(b))
    print("  \t+B\t-B    ")
    print("+A\t{}\t{}\t{}".format(f11, f10, f1p))
    print("-A\t{}\t{}\t{}".format(f01, f00, f0p))
    print("  \t{}\t{}\t{}".format(fp1, fp0, N))
    return f11, f10, f01, f00


def support(f11, f10, f01, f00):
    N = f11 + f10 + f01 + f00
    return f11 / N


def support_a(f11, f10, f01, f00):
    N = f11 + f10 + f01 + f00
    return (f11 + f10) / N


def support_b(f11, f10, f01, f00):
    N = f11 + f10 + f01 + f00
    return (f11 + f01) / N


def confidence_ab(f11, f10, f01, f00):
    f1p = f11 + f10
    return f11 / f1p


def confidence_ba(f11, f10, f01, f00):
    fp1 = f11 + f01
    return f11 / fp1


def interest_factor(f11, f10, f01, f00):
    N = f11 + f10 + f01 + f00
    f1p = f11 + f10
    fp1 = f11 + f01
    return (N * f11) / (f1p * fp1)


def phi_correlation(f11, f10, f01, f00):
    f1p = f11 + f10
    f0p = f01 + f00
    fp1 = f11 + f01
    fp0 = f10 + f00
    num = (f11 * f00) - (f01 * f10)
    denom = math.sqrt(f1p * fp1 * f0p * fp0)
    if denom == 0:
        return 0.0
    return num / denom


def is_score(f11, f10, f01, f00):
    intfac = interest_factor(f11, f10, f01, f00)
    supp = support(f11, f10, f01, f00)
    return math.sqrt(intfac * supp)


ANSWERS_CSV = "answers.csv"
adf = pd.read_csv(ANSWERS_CSV, index_col=0)
adf.columns = [s.split("_")[0] for s in adf.columns]

BINARY_FEATURES_CSV = "binary.csv"
bdf = pd.read_csv(BINARY_FEATURES_CSV, index_col=1)
del bdf["Unnamed: 0"]


def get_rater_rules(rater_idx, consolation=False, min_support=0.5):
    jdf = bdf.copy()
    rates = adf[rater_idx : (rater_idx + 1)].T[rater_idx]
    rates_inv = rates.apply(lambda x: 1 if not x else 0)
    jdf["accepted"] = rates
    jdf["rejected"] = rates_inv

    rq = apriori(jdf, min_support=min_support, max_len=2, use_colnames=True)
    rq["length"] = [len(s) for s in rq["itemsets"]]
    rq = rq.query("length == 2")
    rq["accepting"] = [1 if "accepted" in s else 0 for s in rq["itemsets"]]
    rq["rejecting"] = [1 if "rejected" in s else 0 for s in rq["itemsets"]]

    metrics_list = [
        "a",
        "b",
        "support",
        "support(a)",
        "support(b)",
        "confidence(a -> b)",
        "confidence(b -> a)",
        "interest(a, b)",
        "phi(a, b)",
        "is(a, b)",
    ]

    def get_metric(df, metric):
        def compute(its):
            ct = contingency_table(df, its)
            return metric(*ct)

        return compute

    rq["a"] = [unpack_dual_itemset(s)[0] for s in rq["itemsets"]]
    rq["b"] = [unpack_dual_itemset(s)[1] for s in rq["itemsets"]]
    rq["support(a)"] = rq["itemsets"].apply(get_metric(jdf, support_a))
    rq["support(b)"] = rq["itemsets"].apply(get_metric(jdf, support_b))
    rq["confidence(a -> b)"] = rq["itemsets"].apply(get_metric(jdf, confidence_ab))
    rq["confidence(b -> a)"] = rq["itemsets"].apply(get_metric(jdf, confidence_ba))
    rq["interest(a, b)"] = rq["itemsets"].apply(get_metric(jdf, interest_factor))
    rq["phi(a, b)"] = rq["itemsets"].apply(get_metric(jdf, phi_correlation))
    rq["is(a, b)"] = rq["itemsets"].apply(get_metric(jdf, is_score))
    rqt = rq.query("accepting == 1 or rejecting == 1")
    rj = rqt[metrics_list].sort_values(by=["support", "is(a, b)"], ascending=False)
    if (not consolation) and (len(rj) < 10):
        return get_rater_rules(rater_idx, consolation=True, min_support=0.25)
    output = rj.to_json(orient="records")
    with open("rules/rater{}.json".format(rater_idx), "w") as file:
        json.dump(output, file)
    return output
