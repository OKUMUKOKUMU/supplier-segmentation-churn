"""Supplier feature engineering shared by the notebook (kept in one module so it is testable).

Everything is computed using ONLY data observable on or before the snapshot date (CUTOFF):
  * deliveries in weeks ending on or before CUTOFF
  * payments whose paid_date is on or before CUTOFF (June's milk is paid in July, so it is NOT used)
  * competitor prices for the same months as those payments
  * visits and credit advances dated on or before CUTOFF
The churn label looks only at the 13 weeks AFTER the snapshot. This separation prevents leakage.

The R script (R/supplier_segmentation_churn.R) re-implements every definition below from the raw
files; the two implementations are cross-checked feature by feature.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2026-06-28")      # a Sunday: the last complete collection week of June 2026
ACTIVE_DAYS = 28                         # "active" = delivered in the 4 weeks up to the snapshot
GRACE_DAYS = 28                          # churn must START within 4 weeks of the snapshot ...
SILENT_DAYS = 63                         # ... and last at least 63 days (9 weekly cycles, ~60 days)


def load():
    sup = pd.read_csv(ROOT / "data" / "suppliers.csv", parse_dates=["registered_date"])
    dl = pd.read_csv(ROOT / "data" / "deliveries_weekly.csv.gz", parse_dates=["week_ending"], keep_default_na=False,
                     dtype={"rejection_reason": str})
    pay = pd.read_csv(ROOT / "data" / "payments_monthly.csv", parse_dates=["due_date", "paid_date"])
    mkt = pd.read_csv(ROOT / "data" / "market_prices.csv")
    vis = pd.read_csv(ROOT / "data" / "extension_visits.csv", parse_dates=["visit_date"])
    cred = pd.read_csv(ROOT / "data" / "input_credit.csv", parse_dates=["advance_date"])
    return sup, dl, pay, mkt, vis, cred


def _weekly_matrix(past, ids, cutoff, n):
    """suppliers x n matrix of weekly litres (0 = no delivery) for the n weeks ending at cutoff (col 0 = oldest)."""
    weeks = pd.date_range(end=cutoff, periods=n, freq="7D")
    w = past[past.week_ending >= weeks[0]].pivot_table(index="supplier_id", columns="week_ending",
                                                       values="litres_delivered", aggfunc="sum")
    return w.reindex(index=ids, columns=weeks).fillna(0.0).to_numpy(), weeks


def build_features(sup, dl, pay, mkt, vis, cred, cutoff=CUTOFF):
    past = dl[dl.week_ending <= cutoff]
    ids = np.sort(past.supplier_id.unique())
    g = past.groupby("supplier_id")
    f = pd.DataFrame(index=pd.Index(ids, name="supplier_id"))
    first_week = g.week_ending.min()
    f["recency_days"] = (cutoff - g.week_ending.max()).dt.days
    f["weeks_since_first_delivery"] = (cutoff - first_week).dt.days // 7 + 1

    def win(days):
        return past[past.week_ending > cutoff - pd.Timedelta(days=days)].groupby("supplier_id")

    f["litres_365d"] = win(364).litres_delivered.sum()
    f["litres_13w"] = win(91).litres_delivered.sum()
    f["days_delivered_13w"] = win(91).days_delivered.sum()
    f = f.fillna({"litres_365d": 0, "litres_13w": 0, "days_delivered_13w": 0})

    # observable weeks inside each window (a supplier who joined 5 weeks ago has 5 observable weeks, not 13)
    obs = lambda n: np.minimum(n, f.weeks_since_first_delivery)
    f["litres_per_day_52w"] = f.litres_365d / (7 * obs(52))
    f["delivery_day_share_13w"] = f.days_delivered_13w / (7 * obs(13))
    f["avg_daily_litres_13w"] = np.where(f.days_delivered_13w > 0, f.litres_13w / f.days_delivered_13w.clip(lower=1), 0)

    # consistency: coefficient of variation of weekly litres over the last 26 observable weeks
    M, weeks = _weekly_matrix(past, ids, cutoff, 26)
    mask = (weeks.values[None, :] >= first_week.reindex(ids).values[:, None]).astype(float)
    n = mask.sum(1)
    mean = (M * mask).sum(1) / n
    sd = np.sqrt(((M - mean[:, None]) ** 2 * mask).sum(1) / np.maximum(n - 1, 1))
    f["litres_cv_26w"] = np.where((n >= 2) & (mean > 0), sd / np.where(mean > 0, mean, 1), 0)

    # trend: average weekly litres in the last 8 weeks vs the 18 weeks before, divided by the same ratio for
    # the whole centre (removes the seasonal flush/dry swing every supplier at that centre experiences)
    last8 = (M[:, -8:] * mask[:, -8:]).sum(1) / np.maximum(mask[:, -8:].sum(1), 1)
    n_prior = mask[:, :-8].sum(1)
    prior = (M[:, :-8] * mask[:, :-8]).sum(1) / np.maximum(n_prior, 1)
    centre = sup.set_index("supplier_id").centre_id.reindex(ids).values
    c_ratio = pd.Series(M[:, -8:].sum(1) / 8, index=ids).groupby(centre).sum() / \
        pd.Series(M[:, :-8].sum(1) / 18, index=ids).groupby(centre).sum()
    raw = np.where((n_prior > 0) & (prior > 0), last8 / np.where(prior > 0, prior, 1), 1.0)
    f["litres_trend"] = np.clip(raw / c_ratio.reindex(centre).values, .1, 3)

    # quality (last 26 weeks): litre-weighted fat & SNF, rejection rate and rejection reasons
    q = past[past.week_ending > cutoff - pd.Timedelta(days=182)].copy()
    q["fat_l"], q["snf_l"] = q.fat_pct * q.litres_delivered, q.snf_pct * q.litres_delivered
    qg = q.groupby("supplier_id")
    f["fat_pct"] = qg.fat_l.sum() / qg.litres_delivered.sum()
    f["snf_pct"] = qg.snf_l.sum() / qg.litres_delivered.sum()
    f["rejection_rate"] = qg.litres_rejected.sum() / qg.litres_delivered.sum()
    f["rejection_weeks_26w"] = qg.litres_rejected.apply(lambda s: (s > 0).sum())
    f["antibiotic_26w"] = q[q.rejection_reason == "Antibiotic residue"].groupby("supplier_id").size()
    f["adulteration_26w"] = q[q.rejection_reason == "Adulteration (added water)"].groupby("supplier_id").size()
    f = f.fillna({"rejection_weeks_26w": 0, "antibiotic_26w": 0, "adulteration_26w": 0})
    f["antibiotic_26w"] = (f.antibiotic_26w > 0).astype(int)
    f["adulteration_26w"] = (f.adulteration_26w > 0).astype(int)

    # price & payment: the last 3 monthly payments that had actually been PAID by the snapshot
    p = pay[pay.paid_date <= cutoff].sort_values(["supplier_id", "month"]).groupby("supplier_id").tail(3)
    p = p.merge(sup[["supplier_id", "county"]], on="supplier_id").merge(mkt, on=["county", "month"], how="left",
                                                                         suffixes=("", "_mkt"))
    pg = p.groupby("supplier_id")
    f["price_realised_kes"] = pg.gross_kes.sum() / pg.litres_accepted.sum()
    f["competitor_price_kes"] = pg.competitor_price_kes.mean()
    f["price_gap_kes"] = f.competitor_price_kes - f.price_realised_kes        # > 0: competitors pay more
    f["payment_delay_days"] = pg.payment_delay_days.mean()

    # extension / vet visits and quality training in the last 180 days
    v = vis[(vis.visit_date <= cutoff) & (vis.visit_date > cutoff - pd.Timedelta(days=180))]
    f["advisory_visits_180d"] = v[v.visit_type != "Quality training"].groupby("supplier_id").size()
    f["quality_training_180d"] = v[v.visit_type == "Quality training"].groupby("supplier_id").size()
    f = f.fillna({"advisory_visits_180d": 0, "quality_training_180d": 0})

    # input credit: advances to date minus deductions already recovered from payments
    adv = cred[cred.advance_date <= cutoff].groupby("supplier_id").amount_kes.sum()
    ded = pay[pay.paid_date <= cutoff].groupby("supplier_id").credit_deduction_kes.sum()
    f["credit_balance_kes"] = (adv.reindex(ids).fillna(0) - ded.reindex(ids).fillna(0)).clip(lower=0)
    f["has_credit"] = (f.credit_balance_kes > 0).astype(int)

    f = f.reset_index().merge(sup, on="supplier_id", how="left")
    # suppliers who joined recently may not have been paid yet: use their centre's median (and flag it)
    f["no_payment_yet"] = f.price_gap_kes.isna().astype(int)
    for c in ("price_realised_kes", "competitor_price_kes", "price_gap_kes", "payment_delay_days"):
        f[c] = f[c].fillna(f.groupby("centre_id")[c].transform("median"))
    f["tenure_months"] = ((cutoff - f.registered_date).dt.days / 30.44).round(1)
    f["pays_bank"] = (f.payment_method == "Bank").astype(int)
    return f.drop(columns="registered_date")


def churn_label(dl, cutoff=CUTOFF):
    """1 = no delivery in the 63 days after the 4-week grace period, i.e. the supplier stops within ~4 weeks
    of the snapshot and stays silent for at least 9 weekly cycles (~60 days)."""
    lo, hi = cutoff + pd.Timedelta(days=GRACE_DAYS), cutoff + pd.Timedelta(days=GRACE_DAYS + SILENT_DAYS)
    fut = set(dl.loc[(dl.week_ending > lo) & (dl.week_ending <= hi), "supplier_id"])
    return lambda ids: (~pd.Series(ids).isin(fut)).astype(int).values


def is_test(supplier_id: pd.Series) -> pd.Series:
    """Deterministic 25% test split (supplier number divisible by 4) - identical in Python and R."""
    return supplier_id.str[1:].astype(int) % 4 == 0
