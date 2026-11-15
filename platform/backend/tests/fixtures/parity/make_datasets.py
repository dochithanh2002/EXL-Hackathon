"""Writes the synthetic datasets of the parity fixtures (standard library only, fixed seeds).

    python tests/fixtures/parity/make_datasets.py

The shapes follow the verification models (M2: binomial credit default with recodes; M1: Poisson claim frequency
with an exposure offset), but every value is generated: no real data is committed."""
import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).parent


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def credit(n: int = 600) -> None:
    rng = random.Random(2)
    rows = []
    for i in range(1, n + 1):
        limit = rng.choice(range(10_000, 500_001, 10_000))
        sex = rng.choice([1, 2])
        education = rng.choices([0, 1, 2, 3, 4, 5, 6], weights=[2, 30, 40, 20, 3, 3, 2])[0]
        marriage = rng.choices([0, 1, 2, 3], weights=[3, 45, 48, 4])[0]
        age = rng.randint(21, 70)
        pay = rng.choices([-2, -1, 0, 1, 2, 3], weights=[10, 20, 45, 12, 10, 3])[0]
        bill = round(rng.uniform(0, 1.1) * limit)
        eta = -1.2 - limit / 400_000 + 0.15 * (sex == 1) + 0.1 * (education == 3) - 0.005 * (age - 40) + 0.55 * pay
        default = int(rng.random() < 1 / (1 + math.exp(-eta)))
        rows.append([i, limit, sex, education, marriage, age, pay, bill, default])
    write(HERE / "glm_binomial" / "credit.csv",
          ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "BILL_AMT1",
           "default payment next month"], rows)


def policies(n: int = 800) -> None:
    rng = random.Random(1)
    rows = []
    for i in range(1, n + 1):
        exposure = round(rng.uniform(0.05, 1.0), 2)
        area = rng.choices("ABCDEF", weights=[15, 20, 28, 22, 10, 5])[0]
        power = rng.randint(4, 15)
        veh_age = rng.randint(0, 20)
        driv_age = rng.randint(18, 90)
        bonus = rng.choices([50, 55, 60, 68, 76, 85, 95, 100, 120, 156], weights=[40, 8, 8, 7, 7, 6, 6, 8, 6, 4])[0]
        density = rng.randint(1, 27_000)
        rate = math.exp(-2.6 + 0.04 * (power - 6) - 0.02 * min(veh_age, 12) - 0.006 * (min(driv_age, 80) - 45)
                        + 0.012 * (bonus - 50) + 0.05 * math.log(density) + 0.05 * "ABCDEF".index(area))
        mu, claims, p = rate * exposure, 0, rng.random()
        term = cum = math.exp(-mu)
        while p > cum:
            claims += 1
            term *= mu / claims
            cum += term
        rows.append([i, claims, exposure, area, power, veh_age, driv_age, bonus, density])
    write(HERE / "glm_poisson_offset" / "policies.csv",
          ["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"], rows)


if __name__ == "__main__":
    credit()
    policies()
