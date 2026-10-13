"""Generate synthetic dairy MILK-SUPPLIER data for supplier segmentation and churn analysis.

Context: a Kenyan dairy processor collects raw milk from ~6,000 registered suppliers (about 2,300-2,800 active at any one time) through 10
collection centres in 8 counties. Suppliers are smallholder farmers, farmer groups /
cooperatives that bulk their members' milk, and commercial farms. Competing processors
and informal brokers ("hawkers") buy milk at the farm gate, so a supplier who is unhappy
with price, payment speed or service can simply start selling elsewhere ("side-selling").

How the data is built (so there is real structure to discover):
  * Each supplier has a type, herd size, centre, distance to the centre, payment method,
    and hidden traits: milk hygiene, delivery reliability, breed (fat %) and loyalty.
  * Milk volumes follow Kenya's two rainy seasons: a flush in Apr-Jun and Nov-Dec, and
    dry spells in Jan-Mar and Jul-Sep when milk is scarce and competitors bid up prices.
  * Every week each supplier has a hazard of deciding to leave. The hazard rises with the
    competitor price gap, payment delays, recent rejections and distance, and falls with
    extension/vet visits, an outstanding input-credit (feeds) advance, farmer-group
    membership and longer tenure. About half of leavers stop abruptly; the rest FADE
    (litres and delivery days drop over 2-6 weeks while they side-sell) before stopping.
  * Small herds sometimes pause deliveries for a few weeks (cow drying off, sickness).
    These pauses are NOT churn and add realistic noise.

Outputs (only observable data is exported; the hidden traits are not):
  data/suppliers.csv               one row per supplier (static attributes)
  data/deliveries_weekly.csv.gz    one row per supplier-week with deliveries (gzip to stay small)
  data/payments_monthly.csv        one row per supplier-month paid (price, bonuses, deductions, delay)
  data/market_prices.csv           our base price and surveyed competitor farm-gate price, county x month
  data/extension_visits.csv        one row per extension / vet / quality-training visit
  data/input_credit.csv            one row per input-credit advance (dairy meal, minerals, AI)

All data is randomly generated. No real company, farmer or price data is included.
Usage: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260628)
OUT = Path(__file__).parent
N = 6000
WEEKS = pd.date_range("2025-01-05", "2026-09-27", freq="W-SUN")          # week-ending Sundays (91 weeks)
T = len(WEEKS)
MONTHS = pd.period_range("2025-01", "2026-09", freq="M")
EXTRACT = pd.Timestamp("2026-10-05")                                     # data extract date

# centre: (name, county, mean payment delay in days, extension coverage per week, has cooling tank)
CENTRES = {
    "CC01": ("Ol Kalou",   "Nyandarua",       2, .030, 1), "CC02": ("Kinangop",  "Nyandarua",       4, .022, 1),
    "CC03": ("Molo",       "Nakuru",          9, .012, 0), "CC04": ("Kipkelion", "Kericho",         6, .015, 0),
    "CC05": ("Nyahururu",  "Laikipia",        3, .025, 1), "CC06": ("Limuru",    "Kiambu",          5, .020, 1),
    "CC07": ("Githunguri", "Kiambu",          3, .028, 1), "CC08": ("Iten",      "Elgeyo-Marakwet", 12, .010, 0),
    "CC09": ("Kapsabet",   "Nandi",           7, .014, 0), "CC10": ("Bomet",     "Bomet",           10, .010, 0),
}
CENTRE_SHARE = np.array([.13, .11, .12, .09, .10, .08, .11, .08, .10, .08])
# competing-buyer pressure: extra KES/litre competitors pay above our base price (Kiambu = Nairobi brokers)
COUNTY_PREMIUM = {"Nyandarua": .3, "Nakuru": 1.6, "Kericho": 1.0, "Laikipia": .5, "Kiambu": 2.4,
                  "Elgeyo-Marakwet": 1.2, "Nandi": 1.4, "Bomet": 1.8}
SEASON = {1: .88, 2: .80, 3: .84, 4: 1.00, 5: 1.18, 6: 1.15, 7: 1.02, 8: .93, 9: .88, 10: .94, 11: 1.10, 12: 1.06}
DRY = {1: 1.6, 2: 2.4, 3: 1.8, 4: .4, 5: -.4, 6: .2, 7: 2.0, 8: 2.8, 9: 2.4, 10: 1.0, 11: -.3, 12: 0.0}  # extra competitor premium
TYPES = {  # share, base monthly hazard, cows (lognormal median, sigma), litres/cow/day
    "Smallholder":              (.84, .024, 3.0, .55, 6.5),
    "Farmer group/cooperative": (.05, .007, 80, .60, 7.0),
    "Commercial farm":          (.11, .017, 25, .70, 16.0),
}

# ---------------------------------------------------------------- market prices (county x month)
base_price = {m: 46.0 + (2.0 if m.year == 2026 else 0) + (1.0 if m.month in (1, 2, 3, 8, 9) else 0) for m in MONTHS}
mp = []
for county, prem in COUNTY_PREMIUM.items():
    for m in MONTHS:
        comp = base_price[m] + prem + DRY[m.month] + RNG.normal(0, .5)
        mp.append((county, str(m), base_price[m], round(comp, 1)))
market = pd.DataFrame(mp, columns=["county", "month", "base_price_kes", "competitor_price_kes"])
comp_lookup = {(c, m): p for c, m, _, p in mp}

# ---------------------------------------------------------------- suppliers
type_names = list(TYPES)
cid = list(CENTRES)
s_type = RNG.choice(type_names, N, p=[TYPES[t][0] for t in type_names])
s_centre = RNG.choice(cid, N, p=CENTRE_SHARE / CENTRE_SHARE.sum())
county = np.array([CENTRES[c][1] for c in s_centre])
MIN_HERD = {"Smallholder": 1, "Farmer group/cooperative": 20, "Commercial farm": 10}
herd = np.array([max(MIN_HERD[t], round(RNG.lognormal(np.log(TYPES[t][2]), TYPES[t][3]))) for t in s_type])
yield_cow = np.array([TYPES[t][4] for t in s_type]) * RNG.lognormal(0, .18, N)
dist = np.where(s_type == "Smallholder", RNG.gamma(2.0, 2.6, N),
                np.where(s_type == "Commercial farm", RNG.gamma(2.5, 5.0, N), RNG.gamma(3.0, 5.0, N)))
dist = np.round(np.clip(dist, .3, 45), 1)
pay_bank = RNG.random(N) < np.where(s_type == "Smallholder", .18, .75)
group_member = np.where(s_type == "Farmer group/cooperative", 1, np.where(s_type == "Smallholder", RNG.random(N) < .35, 0)).astype(int)
# registration: 58% registered before the window (up to 8 years earlier), the rest join during it
pre = RNG.random(N) < .58
join_week = np.where(pre, 0, RNG.integers(1, T - 14, N))
reg_date = np.where(pre, WEEKS[0] - pd.to_timedelta(RNG.integers(7, 8 * 365, N), unit="D"),
                    WEEKS[np.minimum(join_week, T - 1)] - pd.to_timedelta(RNG.integers(0, 7, N), unit="D"))
reg_date = pd.to_datetime(reg_date)
tenure0_w = np.where(pre, (WEEKS[0] - reg_date).days / 7, 0)

# hidden traits (never exported)
hygiene = RNG.beta(1.3, 45, N) * np.where(s_type == "Smallholder", 1.3, .7)          # weekly rejection propensity
reliab = RNG.beta(14, 1.4, N) * np.where(s_type == "Smallholder", 1, 1.0)            # chance of delivering on a given day
reliab = np.clip(np.where(s_type == "Smallholder", reliab, np.maximum(reliab, .9)), .45, .995)
fat_mu = np.clip(RNG.normal(3.75, .25, N) + np.where(s_type == "Commercial farm", -.15, 0), 3.1, 4.8)
snf_mu = np.clip(RNG.normal(8.55, .14, N), 8.0, 9.0)
loyalty = RNG.lognormal(0, .35, N)                                                   # unobserved: hazard multiplier
engage = RNG.lognormal(0, .5, N)                                                     # unobserved: visit / credit propensity
q_bonus = np.clip(np.round(4 * (fat_mu - 3.5), 1), -1.5, 3.0)                        # KES/litre (approx. from breed)
v_bonus = np.select([s_type == "Farmer group/cooperative", s_type == "Commercial farm"], [1.5, 1.0], 0.0)
base_h = np.array([TYPES[t][1] for t in s_type])
c_delay = np.array([CENTRES[c][2] for c in s_centre])
c_cover = np.array([CENTRES[c][3] for c in s_centre])
c_cool = np.array([CENTRES[c][4] for c in s_centre])

# ---------------------------------------------------------------- weekly simulation (vectorised across suppliers)
stopped = np.zeros(N, bool)
fade_start = np.full(N, -1)
fade_len = np.zeros(N, int)
pause_left = np.zeros(N, int)
rej_ewma = np.zeros(N)
last_visit_w = np.full(N, -99)
credit_bal = np.zeros(N)
last_delay = c_delay.astype(float)
m_litres = np.zeros(N); m_acc = np.zeros(N); m_fat = np.zeros(N)
deliv, visits, credits, pays = [], [], [], []
decided_week = np.full(N, -1)

for t, wk in enumerate(WEEKS):
    m = wk.to_period("M")
    alive = (join_week <= t) & ~stopped
    # ---- churn decision
    gap = np.array([comp_lookup[(c, str(m))] for c in county]) - (base_price[m] + q_bonus + v_bonus)
    tenure_w = tenure0_w + (t - join_week)
    h_month = (base_h * np.exp(.42 * gap) * np.exp(.08 * last_delay) * (1 + 18.0 * rej_ewma) * (1 + .05 * dist)
               * np.where(t - last_visit_w <= 13, .45, 1.0) * np.where(credit_bal > 0, .45, 1.0)
               * np.where(group_member == 1, .6, 1.0) * np.where(tenure_w < 26, 1.6, 1.0) * loyalty)
    h_week = 1 - (1 - np.clip(h_month, 0, .9)) ** (1 / 4.33)
    new_dec = alive & (decided_week < 0) & (RNG.random(N) < h_week)
    decided_week[new_dec] = t
    abrupt = new_dec & (RNG.random(N) < .4)
    stopped[abrupt] = True
    fading = new_dec & ~abrupt
    fade_start[fading] = t
    fade_len[fading] = RNG.integers(2, 7, fading.sum())
    k = t - fade_start
    fade_mult = np.where(fade_start >= 0, np.clip(1 - (k + 1) / (fade_len + 1), 0, 1), 1.0)
    stopped |= (fade_start >= 0) & (k >= fade_len)
    alive = (join_week <= t) & ~stopped
    # ---- pauses (small herds: cow drying off / sick) - not churn
    start_p = alive & (pause_left == 0) & (herd <= 3) & (RNG.random(N) < .006)
    pause_left[start_p] = RNG.integers(2, 8, start_p.sum())
    delivering = alive & (pause_left == 0)
    pause_left = np.maximum(pause_left - 1, 0)
    # ---- deliveries
    p_day = reliab * np.where(fade_mult < 1, .55 + .45 * fade_mult, 1)
    days = RNG.binomial(7, np.clip(p_day, 0, 1)) * delivering
    daily = herd * yield_cow * SEASON[m.month] * RNG.lognormal(0, .10, N) * np.where(fade_mult < 1, .5 + .5 * fade_mult, 1)
    litres = np.round(days * daily * RNG.lognormal(0, .08, N), 1)
    fat = np.round(fat_mu + RNG.normal(0, .10, N) - .1 * (SEASON[m.month] - 1), 2)
    adult = RNG.random(N) < hygiene * .35
    snf = np.round(snf_mu + RNG.normal(0, .06, N) - .45 * adult, 2)
    hot = m.month in (1, 2, 3, 10)
    p_rej = np.clip(hygiene * (1 + dist / 15) * (1.4 if hot else 1) * (1.6 - .6 * c_cool) + .25 * adult, 0, .8)
    rej = (days > 0) & (RNG.random(N) < p_rej)
    reason = np.where(adult, "Adulteration (added water)",
                      np.where(RNG.random(N) < .35 + dist / 60 + .15 * (1 - c_cool), "Acidity (sour milk)", "Antibiotic residue"))
    rej_l = np.where(rej, np.round(np.minimum(litres, litres / np.maximum(days, 1) * RNG.integers(1, 3, N)), 1), 0)
    rej_ewma = np.where(days > 0, .9 * rej_ewma + .1 * np.where(litres > 0, rej_l / np.maximum(litres, 1e-9), 0), rej_ewma)
    idx = np.flatnonzero(days > 0)
    deliv.append(pd.DataFrame({"supplier_idx": idx, "week_ending": wk, "days_delivered": days[idx], "litres_delivered": litres[idx],
                               "litres_rejected": rej_l[idx], "rejection_reason": np.where(rej[idx], reason[idx], ""),
                               "fat_pct": fat[idx], "snf_pct": snf[idx]}))
    m_litres += litres; m_acc += litres - rej_l; m_fat += fat * (litres - rej_l)
    # ---- extension / vet visits (centre coverage x engagement; quality training after a rejection)
    pv = c_cover * engage * np.where(s_type == "Smallholder", 1, 2.2)
    v_reg = alive & (RNG.random(N) < pv)
    v_q = alive & rej & (RNG.random(N) < .30)
    for mask, vt in ((v_reg, None), (v_q, "Quality training")):
        j = np.flatnonzero(mask)
        if len(j):
            types = RNG.choice(["Extension officer", "Vet / AI service"], len(j), p=[.6, .4]) if vt is None else np.full(len(j), vt)
            visits.append(pd.DataFrame({"supplier_idx": j, "visit_date": wk - pd.to_timedelta(RNG.integers(0, 7, len(j)), unit="D"),
                                        "visit_type": types}))
    last_visit_w[v_reg | v_q] = t
    # ---- input-credit advances (dairy meal, minerals, AI) - more likely for engaged, reliable suppliers
    pc = .025 * engage * (reliab ** 4) * np.where(fade_mult < 1, .2, 1)
    cr = alive & (RNG.random(N) < pc) & (credit_bal == 0)
    j = np.flatnonzero(cr)
    if len(j):
        amt = np.round(np.where(s_type[j] == "Smallholder", RNG.uniform(3000, 15000, len(j)), RNG.uniform(15000, 90000, len(j))), -2)
        credit_bal[j] += amt
        credits.append(pd.DataFrame({"supplier_idx": j, "advance_date": wk - pd.to_timedelta(RNG.integers(0, 7, len(j)), unit="D"),
                                     "item": RNG.choice(["Dairy meal", "Mineral lick", "AI service", "Fodder seed"], len(j), p=[.6, .15, .15, .1]),
                                     "amount_kes": amt}))
    # ---- month end: compute payment
    if t == T - 1 or WEEKS[t + 1].to_period("M") != m:
        j = np.flatnonzero(m_acc > 0)
        fat_m = m_fat[j] / m_acc[j]
        qb = np.clip(np.round(4 * (fat_m - 3.5), 1), -1.5, 3.0)
        price = base_price[m] + qb + v_bonus[j]
        gross = np.round(m_acc[j] * price, 0)
        ded = np.round(np.minimum(credit_bal[j], .25 * gross), 0)   # recover up to 25% of the cheque
        credit_bal[j] -= ded
        delay = np.maximum(0, np.round(c_delay[j] + 3 * pay_bank[j] + RNG.normal(0, 3, len(j)) + RNG.exponential(1.5, len(j)))).astype(int)
        last_delay[j] = delay
        due = (m + 1).to_timestamp() + pd.Timedelta(days=9)            # due on the 10th of the following month
        pays.append(pd.DataFrame({"supplier_idx": j, "month": str(m), "litres_accepted": np.round(m_acc[j], 1),
                                  "base_price_kes": base_price[m], "quality_bonus_kes": qb, "volume_bonus_kes": v_bonus[j],
                                  "price_per_litre_kes": np.round(price, 2), "gross_kes": gross, "credit_deduction_kes": ded,
                                  "net_paid_kes": gross - ded, "due_date": due,
                                  "paid_date": due + pd.to_timedelta(delay, unit="D"), "payment_delay_days": delay}))
        m_litres[:] = 0; m_acc[:] = 0; m_fat[:] = 0

# ---------------------------------------------------------------- export
sid = np.array([f"S{i:05d}" for i in range(1, N + 1)])
sup = pd.DataFrame({"supplier_id": sid, "supplier_type": s_type, "centre_id": s_centre,
                    "centre_name": [CENTRES[c][0] for c in s_centre], "county": county,
                    "registered_date": reg_date.strftime("%Y-%m-%d"), "herd_size_cows": herd, "distance_km": dist,
                    "payment_method": np.where(pay_bank, "Bank", "M-Pesa"), "group_member": group_member})


def fin(frames, date_cols):
    d = pd.concat(frames, ignore_index=True)
    d.insert(0, "supplier_id", sid[d.pop("supplier_idx").values])
    for c in date_cols:
        d[c] = pd.to_datetime(d[c]).dt.strftime("%Y-%m-%d")
    return d.sort_values(["supplier_id", date_cols[0]]).reset_index(drop=True)


deliv = fin(deliv, ["week_ending"])
visits = fin(visits, ["visit_date"])
credits = fin(credits, ["advance_date"])
pays = fin(pays, ["due_date", "paid_date"])
pays = pays[["supplier_id", "month"] + [c for c in pays.columns if c not in ("supplier_id", "month")]]
pays = pays[pd.to_datetime(pays.due_date) <= EXTRACT].sort_values(["supplier_id", "month"])  # Sept not yet due

sup.to_csv(OUT / "suppliers.csv", index=False)
deliv.to_csv(OUT / "deliveries_weekly.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
pays.to_csv(OUT / "payments_monthly.csv", index=False)
market.to_csv(OUT / "market_prices.csv", index=False)
visits.to_csv(OUT / "extension_visits.csv", index=False)
credits.to_csv(OUT / "input_credit.csv", index=False)
print(f"{N:,} suppliers | {len(deliv):,} supplier-weeks | {deliv.litres_delivered.sum()/1e6:,.1f}m litres | "
      f"{len(pays):,} payments | {len(visits):,} visits | {len(credits):,} credit advances")
# hidden ground truth, printed for calibration only (not saved)
print("suppliers who decided to leave during the window:", (decided_week >= 0).sum())
