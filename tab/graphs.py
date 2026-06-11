import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

env_training = pd.read_csv("../data/environment_training.csv")
corr_training = pd.read_csv("../data/corrosions_training.csv")

corr_training["observation_date"] = pd.to_datetime(corr_training["observation_date"])
corr_training = corr_training.assign(
    months=(
        (corr_training["observation_date"].dt.year - corr_training["aircraft_delivery_year"]) * 12
        + (corr_training["observation_date"].dt.month - corr_training["aircraft_delivery_month"])
    )
)

res = pd.merge(env_training, corr_training, on="aircraft_id", how="inner")

# Per-aircraft aggregates (one row per aircraft)
per_ac = res.groupby("aircraft_id").agg(
    months=("months", "first"),
    mean_humidity=("metar_relative_humidity", "mean"),
    mean_temp=("metar_temperature_c", "mean"),
    mean_salt_fine=("sea_salt_aerosol_05_5_mixing_ratio", "mean"),
    mean_salt_coarse=("sea_salt_aerosol_5_20_mixing_ratio", "mean"),
    mean_so2=("sulphur_dioxide_mass_mixing_ratio", "mean"),
    mean_parking=("total_parking_minutes", "mean"),
    mean_no2=("nitrogen_dioxide_mass_mixing_ratio", "mean"),
    mean_precipitation=("metar_hour_precipitation", "mean"),
    mean_dust=("dust_aerosol_09_20_mixing_ratio", "mean"),
    n_obs=("year_month", "count"),
).reset_index()

per_ac["total_salt"] = per_ac["mean_salt_fine"] + per_ac["mean_salt_coarse"]

# Calendar-month aggregates for seasonality
res["cal_month"] = pd.to_datetime(res["month_start_date"]).dt.month
monthly_seasonal = res.groupby("cal_month").agg(
    humidity=("metar_relative_humidity", "mean"),
    temp=("metar_temperature_c", "mean"),
    salt_fine=("sea_salt_aerosol_05_5_mixing_ratio", "mean"),
    salt_coarse=("sea_salt_aerosol_5_20_mixing_ratio", "mean"),
    precipitation=("metar_hour_precipitation", "mean"),
).reset_index()

MONTHS_LABEL = "Months to first corrosion"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Correlation of all per-aircraft numeric features with months ──
corr_with_months = (
    per_ac.select_dtypes(include="number")
    .corr()["months"]
    .drop("months")
    .sort_values()
)

# ── Months quartile labels (for colouring) ──
per_ac["months_q"] = pd.qcut(per_ac["months"], 4, labels=["Q1 (fast)", "Q2", "Q3", "Q4 (slow)"])

fig = plt.figure(figsize=(22, 30))
fig.suptitle("Aircraft Corrosion Dataset — 10 Insightful Graphs", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

# ────────────────────────────────────────────────
# 1. Distribution of months to first corrosion
# ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(per_ac["months"], bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax1.axvline(per_ac["months"].median(), color="tomato", linestyle="--", linewidth=1.5, label=f"Median {per_ac['months'].median():.0f} mo")
ax1.set_xlabel(MONTHS_LABEL)
ax1.set_ylabel("Number of aircraft")
ax1.set_title("1 · Distribution of Corrosion Onset Time")
ax1.legend()

# ────────────────────────────────────────────────
# 2. Correlation bar chart — what drives months?
# ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
colors = ["#D94F3D" if v < 0 else "#4C72B0" for v in corr_with_months]
ax2.barh(corr_with_months.index, corr_with_months.values, color=colors)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Pearson r  with  months-to-corrosion")
ax2.set_title("2 · Feature Correlation with Corrosion Onset")
ax2.tick_params(axis="y", labelsize=8)

# ────────────────────────────────────────────────
# 3. Sea-salt (coarse) vs months  — strongest predictor
# ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
slope, intercept, r, p, _ = stats.linregress(per_ac["mean_salt_coarse"], per_ac["months"])
x_line = np.linspace(per_ac["mean_salt_coarse"].min(), per_ac["mean_salt_coarse"].max(), 200)
ax3.scatter(per_ac["mean_salt_coarse"], per_ac["months"], alpha=0.35, s=18, color="#4C72B0")
ax3.plot(x_line, intercept + slope * x_line, color="tomato", linewidth=2, label=f"r = {r:.2f}")
ax3.set_xlabel("Mean sea-salt aerosol 5–20 µm (coarse)")
ax3.set_ylabel(MONTHS_LABEL)
ax3.set_title("3 · Coarse Sea-Salt Aerosol vs Corrosion Onset")
ax3.legend()

# ────────────────────────────────────────────────
# 4. Sea-salt (fine) vs months
# ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
slope, intercept, r, p, _ = stats.linregress(per_ac["mean_salt_fine"], per_ac["months"])
x_line = np.linspace(per_ac["mean_salt_fine"].min(), per_ac["mean_salt_fine"].max(), 200)
ax4.scatter(per_ac["mean_salt_fine"], per_ac["months"], alpha=0.35, s=18, color="#DD8452")
ax4.plot(x_line, intercept + slope * x_line, color="tomato", linewidth=2, label=f"r = {r:.2f}")
ax4.set_xlabel("Mean sea-salt aerosol 0.5–5 µm (fine)")
ax4.set_ylabel(MONTHS_LABEL)
ax4.set_title("4 · Fine Sea-Salt Aerosol vs Corrosion Onset")
ax4.legend()

# ────────────────────────────────────────────────
# 5. Total sea salt vs months (fine + coarse)
# ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
slope, intercept, r, p, _ = stats.linregress(per_ac["total_salt"], per_ac["months"])
x_line = np.linspace(per_ac["total_salt"].min(), per_ac["total_salt"].max(), 200)
ax5.scatter(per_ac["total_salt"], per_ac["months"], alpha=0.35, s=18, color="#55A868")
ax5.plot(x_line, intercept + slope * x_line, color="tomato", linewidth=2, label=f"r = {r:.2f}")
ax5.set_xlabel("Mean total sea-salt aerosol (fine + coarse)")
ax5.set_ylabel(MONTHS_LABEL)
ax5.set_title("5 · Total Sea-Salt Aerosol vs Corrosion Onset")
ax5.legend()

# ────────────────────────────────────────────────
# 6. Relative humidity vs months
# ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
slope, intercept, r, p, _ = stats.linregress(
    per_ac["mean_humidity"].dropna(),
    per_ac.loc[per_ac["mean_humidity"].notna(), "months"],
)
x_line = np.linspace(per_ac["mean_humidity"].min(), per_ac["mean_humidity"].max(), 200)
ax6.scatter(per_ac["mean_humidity"], per_ac["months"], alpha=0.35, s=18, color="#C44E52")
ax6.plot(x_line, intercept + slope * x_line, color="tomato", linewidth=2, label=f"r = {r:.2f}")
ax6.set_xlabel("Mean relative humidity (%)")
ax6.set_ylabel(MONTHS_LABEL)
ax6.set_title("6 · Relative Humidity vs Corrosion Onset")
ax6.legend()

# ────────────────────────────────────────────────
# 7. Parking time by months quartile (box plot)
# ────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
groups = [per_ac.loc[per_ac["months_q"] == q, "mean_parking"] for q in ["Q1 (fast)", "Q2", "Q3", "Q4 (slow)"]]
bp = ax7.boxplot(groups, patch_artist=True, widths=0.5)
palette = ["#D94F3D", "#DD8452", "#55A868", "#4C72B0"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax7.set_xticklabels(["Q1\n(fast corrosion)", "Q2", "Q3", "Q4\n(slow corrosion)"])
ax7.set_ylabel("Mean monthly parking (minutes)")
ax7.set_title("7 · Parking Time by Corrosion Onset Quartile")

# ────────────────────────────────────────────────
# 8. Seasonal pattern: humidity & temperature
# ────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
ax8b = ax8.twinx()
ax8.bar(monthly_seasonal["cal_month"], monthly_seasonal["humidity"],
        color="#4C72B0", alpha=0.5, label="Rel. humidity (%)")
ax8b.plot(monthly_seasonal["cal_month"], monthly_seasonal["temp"],
          color="tomato", marker="o", linewidth=2, label="Temp (°C)")
ax8.set_xticks(range(1, 13))
ax8.set_xticklabels(MONTH_NAMES)
ax8.set_ylabel("Mean relative humidity (%)", color="#4C72B0")
ax8b.set_ylabel("Mean temperature (°C)", color="tomato")
ax8.set_title("8 · Seasonal Patterns: Humidity & Temperature")
ax8.legend(loc="upper left", fontsize=8)
ax8b.legend(loc="upper right", fontsize=8)

# ────────────────────────────────────────────────
# 9. Seasonal sea-salt aerosol (fine + coarse)
# ────────────────────────────────────────────────
ax9 = fig.add_subplot(gs[4, 0])
ax9.plot(monthly_seasonal["cal_month"], monthly_seasonal["salt_fine"],
         marker="o", color="#DD8452", linewidth=2, label="Fine (0.5–5 µm)")
ax9.plot(monthly_seasonal["cal_month"], monthly_seasonal["salt_coarse"],
         marker="s", color="#55A868", linewidth=2, label="Coarse (5–20 µm)")
ax9.set_xticks(range(1, 13))
ax9.set_xticklabels(MONTH_NAMES)
ax9.set_ylabel("Mean sea-salt aerosol mixing ratio")
ax9.set_title("9 · Seasonal Sea-Salt Aerosol Levels")
ax9.legend()

# ────────────────────────────────────────────────
# 10. Temperature vs Humidity coloured by months quartile
# ────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[4, 1])
q_colors = {"Q1 (fast)": "#D94F3D", "Q2": "#DD8452", "Q3": "#55A868", "Q4 (slow)": "#4C72B0"}
for q, color in q_colors.items():
    sub = per_ac[per_ac["months_q"] == q]
    ax10.scatter(sub["mean_temp"], sub["mean_humidity"], alpha=0.45, s=20,
                 color=color, label=q)
ax10.set_xlabel("Mean temperature (°C)")
ax10.set_ylabel("Mean relative humidity (%)")
ax10.set_title("10 · Temp vs Humidity — Coloured by Corrosion Quartile")
ax10.legend(title="Corrosion onset", fontsize=8)

plt.savefig("corrosion_insights.png", dpi=150, bbox_inches="tight")
print("Saved corrosion_insights.png")
