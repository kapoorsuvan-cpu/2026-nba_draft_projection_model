"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  LineChart as LineIcon,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  Radar,
  RadarChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Prospect = {
  espn_rank: number;
  player_name: string;
  slug: string;
  headshot_url: string;
  position: "G" | "F" | "C";
  espn_position?: string;
  college_team?: string;
  conference?: string;
  games_played?: number;
  minutes_per_game?: number;
  points_per_game?: number;
  rebounds_per_game?: number;
  assists_per_game?: number;
  steals_per_game?: number;
  blocks_per_game?: number;
  usage_rate?: number;
  recruiting_rank?: number;
  recruiting_rating?: number;
  recruiting_stars?: number;
  feature_coverage_pct?: number;
  predicted_label?: string;
  prob_star?: number;
  prob_rotation?: number;
  prob_not_nba_level?: number;
  confidence?: number;
  model_algorithm?: string;
  model_macro_f1?: number;
  model_test_n?: number;
};

type ModelMetric = {
  position: string;
  best_algorithm?: string;
  macro_f1?: number;
  balanced_accuracy?: number;
  accuracy?: number;
  weighted_f1?: number;
  test_n?: number;
  train_n?: number;
  n_features?: number;
};

type FeatureImportance = {
  position: string;
  feature: string;
  importance: number;
  algorithm?: string;
};

type DashboardSummary = {
  trainingSummary?: {
    rows?: number;
    minDraftYear?: number;
    maxDraftYear?: number;
  };
  historicalOutcomeDistribution?: { outcome: string; count: number; share: number }[];
  currentClassOutcomeDistribution?: { outcome: string; count: number; share: number }[];
  historicalOutcomeByPosition?: { position: string; outcome: string; count: number; share: number }[];
  historicalOutcomeByYear?: { draft_year: number; outcome: string; count: number }[];
  currentClassAverageProbabilities?: { outcome: string; probability: number }[];
};

const labelMap: Record<string, string> = {
  Star: "Star",
  Rotation: "Rotation",
  "Not NBA Level": "Not NBA",
};

function pct(v?: number) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function num(v?: number, digits = 1) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function cleanFeatureName(feature: string) {
  return feature
    .replace("num__", "")
    .replace("cat__", "")
    .replaceAll("_", " ");
}

function initials(name?: string) {
  if (!name) return "?";
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((x) => x[0])
    .join("")
    .toUpperCase();
}

function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-3xl border border-slate-200 bg-white p-6 shadow-sm ${className}`}>
      {children}
    </div>
  );
}

function SectionTitle({
  icon,
  title,
  subtitle,
}: {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-5">
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      </div>
      {subtitle && <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{subtitle}</p>}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-[96px] rounded-2xl bg-slate-100 px-5 py-4 text-center">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-950">{value}</div>
    </div>
  );
}

function Headshot({ prospect }: { prospect: Prospect }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="flex h-28 w-28 shrink-0 items-center justify-center rounded-3xl bg-slate-200 text-3xl font-bold text-slate-700">
        {initials(prospect.player_name)}
      </div>
    );
  }

  return (
    <img
      src={prospect.headshot_url}
      alt={prospect.player_name}
      onError={() => setFailed(true)}
      className="h-28 w-28 shrink-0 rounded-3xl object-cover ring-1 ring-slate-200"
    />
  );
}

function outcomeProbabilityData(p?: Prospect) {
  return [
    { outcome: "Star", probability: Math.round((p?.prob_star ?? 0) * 100) },
    { outcome: "Rotation", probability: Math.round((p?.prob_rotation ?? 0) * 100) },
    { outcome: "Not NBA", probability: Math.round((p?.prob_not_nba_level ?? 0) * 100) },
  ];
}

function productionRadarData(p?: Prospect) {
  return [
    { feature: "PPG", value: p?.points_per_game ?? 0 },
    { feature: "RPG", value: p?.rebounds_per_game ?? 0 },
    { feature: "APG", value: p?.assists_per_game ?? 0 },
    { feature: "SPG", value: p?.steals_per_game ?? 0 },
    { feature: "BPG", value: p?.blocks_per_game ?? 0 },
    { feature: "Usage", value: p?.usage_rate ?? 0 },
  ];
}

function distributionCompareData(summary?: DashboardSummary) {
  const labels = ["Star", "Rotation", "Not NBA Level"];

  return labels.map((label) => {
    const h = summary?.historicalOutcomeDistribution?.find((x) => x.outcome === label);
    const c = summary?.currentClassOutcomeDistribution?.find((x) => x.outcome === label);

    return {
      outcome: labelMap[label] ?? label,
      historical: Math.round((h?.share ?? 0) * 100),
      current: Math.round((c?.share ?? 0) * 100),
    };
  });
}

function yearlyStarData(summary?: DashboardSummary) {
  const rows = summary?.historicalOutcomeByYear ?? [];
  const years = Array.from(new Set(rows.map((r) => r.draft_year))).sort();

  return years.map((year) => {
    const yearRows = rows.filter((r) => r.draft_year === year);
    const total = yearRows.reduce((s, r) => s + r.count, 0);
    const star = yearRows.find((r) => r.outcome === "Star")?.count ?? 0;
    const rotation = yearRows.find((r) => r.outcome === "Rotation")?.count ?? 0;

    return {
      year,
      starShare: total ? Math.round((star / total) * 100) : 0,
      nbaShare: total ? Math.round(((star + rotation) / total) * 100) : 0,
    };
  });
}

function positionDistributionData(summary?: DashboardSummary, position = "F") {
  const rows = summary?.historicalOutcomeByPosition ?? [];
  return ["Star", "Rotation", "Not NBA Level"].map((outcome) => {
    const row = rows.find((r) => r.position === position && r.outcome === outcome);
    return {
      outcome: labelMap[outcome] ?? outcome,
      share: Math.round((row?.share ?? 0) * 100),
    };
  });
}

export default function Page() {
  const [prospects, setProspects] = useState<Prospect[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);
  const [importance, setImportance] = useState<FeatureImportance[]>([]);
  const [summary, setSummary] = useState<DashboardSummary>({});
  const [query, setQuery] = useState("");
  const [selectedSlug, setSelectedSlug] = useState("");
  const [manualPosition, setManualPosition] = useState<"G" | "F" | "C">("F");
  const [manual, setManual] = useState({
    points_per_game: 18,
    rebounds_per_game: 6,
    assists_per_game: 3,
    steals_per_game: 1,
    blocks_per_game: 0.7,
    minutes_per_game: 30,
    usage_rate: 25,
    recruiting_rank: 25,
  });

  useEffect(() => {
    async function load() {
      const [p, m, f, s] = await Promise.all([
        fetch("/data/prospects.json").then((r) => r.json()),
        fetch("/data/model_metrics.json").then((r) => r.json()).catch(() => []),
        fetch("/data/feature_importance.json").then((r) => r.json()).catch(() => []),
        fetch("/data/dashboard_summary.json").then((r) => r.json()).catch(() => ({})),
      ]);

      setProspects(p);
      setMetrics(m);
      setImportance(f);
      setSummary(s);

      if (p?.length) setSelectedSlug(p[0].slug);
    }

    load();
  }, []);

  const selected = useMemo(
    () => prospects.find((p) => p.slug === selectedSlug) ?? prospects[0],
    [prospects, selectedSlug]
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return prospects;

    return prospects.filter((p) =>
      [
        p.player_name,
        p.college_team,
        p.conference,
        p.position,
        p.predicted_label,
      ]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }, [query, prospects]);

  const metricsByPosition = useMemo(() => {
    const out: Record<string, ModelMetric> = {};
    for (const m of metrics) out[m.position] = m;
    return out;
  }, [metrics]);

  const selectedImportance = useMemo(() => {
    const pos = selected?.position ?? "F";
    return importance
      .filter((x) => x.position === pos)
      .slice(0, 12)
      .map((x) => ({
        feature: cleanFeatureName(x.feature),
        importance: Number(x.importance),
      }));
  }, [importance, selected]);

  const manualProjection = useMemo(() => {
    const ppg = Number(manual.points_per_game) || 0;
    const rpg = Number(manual.rebounds_per_game) || 0;
    const apg = Number(manual.assists_per_game) || 0;
    const spg = Number(manual.steals_per_game) || 0;
    const bpg = Number(manual.blocks_per_game) || 0;
    const mpg = Number(manual.minutes_per_game) || 0;
    const usage = Number(manual.usage_rate) || 0;
    const rank = Number(manual.recruiting_rank) || 100;

    const rankPrior = Math.max(0, 101 - rank);

    let production = 0;
    if (manualPosition === "G") {
      production = ppg * 1.0 + apg * 2.6 + spg * 3.0 + usage * 0.45;
    } else if (manualPosition === "F") {
      production = ppg * 1.15 + rpg * 1.6 + apg * 1.3 + spg * 2.2 + bpg * 2.2 + usage * 0.55;
    } else {
      production = ppg * 0.95 + rpg * 2.1 + bpg * 4.4 + usage * 0.45;
    }

    const minutesBonus = Math.min(8, mpg / 4);
    const score = Math.max(0, Math.min(100, production + minutesBonus + rankPrior * 0.35));

    let tier = "Low NBA Projection";
    if (score >= 75) tier = "Star Upside";
    else if (score >= 55) tier = "Rotation-or-Better";
    else if (score >= 40) tier = "Fringe / Developmental";

    return { score, tier };
  }, [manual, manualPosition]);

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-10 grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-600">
              <Brain size={16} />
              NBA College Projection Model
            </div>
            <h1 className="max-w-5xl text-4xl font-semibold tracking-tight md:text-6xl">
              2026 NBA draft prospect outcome dashboard
            </h1>
            <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-600">
              Explore projected NBA outcomes from college production, position group, prospect rank,
              and historical draft outcomes. The dashboard emphasizes probabilities, model uncertainty,
              and feature context instead of treating hard labels as scouting truth.
            </p>
          </div>

          <Card>
            <SectionTitle
              icon={<AlertTriangle className="text-amber-500" size={20} />}
              title="Read before interpreting"
            />
            <p className="text-sm leading-6 text-slate-600">
              These are experimental model outputs. Guard performance is weak, center samples are small,
              and hard labels can be misleading when probabilities are close. Use the probability chart,
              model metrics, and feature coverage together.
            </p>
          </Card>
        </div>

        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <Card>
            <div className="text-sm text-slate-500">Training rows</div>
            <div className="mt-1 text-3xl font-semibold">{summary.trainingSummary?.rows ?? "—"}</div>
            <div className="mt-1 text-sm text-slate-500">
              {summary.trainingSummary?.minDraftYear ?? "—"}–{summary.trainingSummary?.maxDraftYear ?? "—"}
            </div>
          </Card>
          {(["G", "F", "C"] as const).map((pos) => {
            const m: ModelMetric = metricsByPosition[pos] ?? { position: pos };
            return (
              <Card key={pos}>
                <div className="text-sm text-slate-500">{pos} model</div>
                <div className="mt-1 text-xl font-semibold">{m.best_algorithm ?? "—"}</div>
                <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <div className="text-slate-500">F1</div>
                    <div className="font-semibold">{num(m.macro_f1, 3)}</div>
                  </div>
                  <div>
                    <div className="text-slate-500">Acc.</div>
                    <div className="font-semibold">{num(m.accuracy, 3)}</div>
                  </div>
                  <div>
                    <div className="text-slate-500">Test n</div>
                    <div className="font-semibold">{m.test_n ?? "—"}</div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>

        <div className="mb-8 grid gap-6 lg:grid-cols-2">
          <Card>
            <SectionTitle
              icon={<BarChart3 size={20} />}
              title="Current class vs historical outcome mix"
              subtitle="Compares the projected 2026 outcome distribution with the historical labeled training set."
            />
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionCompareData(summary)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="outcome" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="historical" name="Historical %" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="current" name="2026 projected %" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <SectionTitle
              icon={<LineIcon size={20} />}
              title="Historical NBA-level rate by draft year"
              subtitle="Share of historical players labeled Star or Rotation."
            />
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={yearlyStarData(summary)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="nbaShare" name="Star + Rotation %" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="starShare" name="Star %" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <Card className="h-fit">
            <SectionTitle icon={<Search size={20} />} title="2026 prospect selector" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search player, school, outcome..."
              className="mb-4 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm outline-none focus:border-slate-500"
            />

            <div className="max-h-[720px] space-y-2 overflow-auto pr-1">
              {filtered.map((p) => (
                <button
                  key={p.slug}
                  onClick={() => setSelectedSlug(p.slug)}
                  className={`flex w-full items-center gap-3 rounded-2xl border p-3 text-left transition ${
                    selectedSlug === p.slug
                      ? "border-slate-950 bg-slate-950 text-white"
                      : "border-slate-200 bg-white hover:border-slate-400"
                  }`}
                >
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-sm font-bold ${
                      selectedSlug === p.slug ? "bg-white text-slate-950" : "bg-slate-100 text-slate-700"
                    }`}
                  >
                    {p.espn_rank}
                  </div>
                  <div className="min-w-0">
                    <div className="truncate font-medium">{p.player_name}</div>
                    <div className={`truncate text-xs ${selectedSlug === p.slug ? "text-slate-300" : "text-slate-500"}`}>
                      {p.position} · {p.college_team ?? "Unknown"} · {p.predicted_label ?? "No prediction"}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {selected && (
            <div className="space-y-6">
              <Card>
                <div className="flex flex-col gap-6 md:flex-row md:items-center">
                  <Headshot prospect={selected} />

                  <div className="flex-1">
                    <div className="text-sm font-medium text-slate-500">
                      Prospect rank #{selected.espn_rank}
                    </div>
                    <h2 className="text-3xl font-semibold tracking-tight">{selected.player_name}</h2>
                    <p className="mt-1 text-slate-600">
                      {selected.position} · {selected.college_team ?? "Unknown"} · {selected.conference ?? "Unknown"}
                    </p>

                    <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
                      <Stat label="PPG" value={num(selected.points_per_game)} />
                      <Stat label="RPG" value={num(selected.rebounds_per_game)} />
                      <Stat label="APG" value={num(selected.assists_per_game)} />
                      <Stat label="SPG" value={num(selected.steals_per_game)} />
                      <Stat label="BPG" value={num(selected.blocks_per_game)} />
                      <Stat label="Usage" value={num(selected.usage_rate)} />
                    </div>
                  </div>

                  <div className="rounded-3xl bg-slate-950 p-6 text-white md:w-72">
                    <div className="text-sm text-slate-300">Projected outcome</div>
                    <div className="mt-1 text-2xl font-semibold">{selected.predicted_label ?? "—"}</div>
                    <div className="mt-5 space-y-2 text-sm text-slate-300">
                      <div>Confidence: {pct(selected.confidence)}</div>
                      <div>Feature coverage: {pct(selected.feature_coverage_pct)}</div>
                      <div>Model: {selected.model_algorithm ?? "—"}</div>
                    </div>
                  </div>
                </div>
              </Card>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <SectionTitle title="Outcome probabilities" />
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={outcomeProbabilityData(selected)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="outcome" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Bar dataKey="probability" name="Probability %" radius={[10, 10, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card>
                  <SectionTitle title="Production profile" />
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={productionRadarData(selected)}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="feature" />
                        <PolarRadiusAxis />
                        <Radar dataKey="value" fillOpacity={0.35} />
                        <Tooltip />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>

              <div className="grid gap-6 lg:grid-cols-2">
                <Card>
                  <SectionTitle
                    title={`Historical outcomes for ${selected.position} prospects`}
                    subtitle="Shows how outcomes are distributed for the selected position group in the training data."
                  />
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={positionDistributionData(summary, selected.position)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="outcome" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="share" name="Historical share %" radius={[10, 10, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>

                <Card>
                  <SectionTitle
                    title={`Top ${selected.position} model feature weights`}
                    subtitle="Feature importance from the best model selected for this position group."
                  />
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={selectedImportance} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis dataKey="feature" type="category" width={170} />
                        <Tooltip />
                        <Bar dataKey="importance" radius={[0, 10, 10, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>
            </div>
          )}
        </div>

        <section className="mt-8 grid gap-6 lg:grid-cols-2">
          <Card>
            <SectionTitle
              icon={<SlidersHorizontal size={20} />}
              title="Custom prospect input"
              subtitle="Enter a position group and feature values to explore how a prospect profile could project."
            />

            <label className="text-sm font-semibold">Position group</label>
            <select
              value={manualPosition}
              onChange={(e) => setManualPosition(e.target.value as "G" | "F" | "C")}
              className="mb-4 mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3"
            >
              <option value="G">Guard</option>
              <option value="F">Forward</option>
              <option value="C">Center</option>
            </select>

            <div className="grid gap-4 md:grid-cols-2">
              {Object.entries(manual).map(([key, value]) => (
                <label key={key} className="text-sm font-semibold capitalize">
                  {key.replaceAll("_", " ")}
                  <input
                    type="number"
                    step="0.1"
                    value={value}
                    onChange={(e) =>
                      setManual((prev) => ({
                        ...prev,
                        [key]: Number(e.target.value),
                      }))
                    }
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3"
                  />
                </label>
              ))}
            </div>
          </Card>

          <Card>
            <SectionTitle icon={<CheckCircle2 size={20} />} title="Manual projection result" />
            <div className="rounded-3xl bg-slate-950 p-8 text-white">
              <div className="text-sm text-slate-300">Projection tier</div>
              <div className="mt-2 text-4xl font-semibold">{manualProjection.tier}</div>
              <div className="mt-8 text-sm text-slate-300">Profile score</div>
              <div className="mt-3 h-4 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-white"
                  style={{ width: `${manualProjection.score}%` }}
                />
              </div>
              <div className="mt-2 text-right text-sm text-slate-300">
                {Math.round(manualProjection.score)} / 100
              </div>
            </div>
            <p className="mt-4 rounded-2xl bg-amber-50 p-4 text-sm leading-6 text-amber-900">
              This section is an interactive frontend estimator. The prospect selector above uses
              the exported model predictions from your Python pipeline. For exact custom-input model
              inference, connect this form to a Python backend API that loads your saved sklearn models.
            </p>
          </Card>
        </section>

        <section className="mt-8">
          <Card>
            <SectionTitle
              title="Process and limitations"
              subtitle="How the dashboard should be interpreted."
            />
            <div className="grid gap-6 text-sm leading-6 text-slate-600 md:grid-cols-4">
              <div>
                <div className="mb-2 font-semibold text-slate-950">1. Historical labels</div>
                <p>
                  Historical drafted players are labeled by NBA outcome using career production and
                  role-level success signals.
                </p>
              </div>
              <div>
                <div className="mb-2 font-semibold text-slate-950">2. College features</div>
                <p>
                  The model uses college scoring, usage, rebounding, playmaking, defensive events, and
                  prospect rank signals.
                </p>
              </div>
              <div>
                <div className="mb-2 font-semibold text-slate-950">3. Position models</div>
                <p>
                  Separate G, F, and C models are used because translation signals differ by role.
                </p>
              </div>
              <div>
                <div className="mb-2 font-semibold text-slate-950">4. Main limitation</div>
                <p>
                  The model is exploratory. Small test samples and close probabilities mean the
                  probability distribution matters more than the single hard label.
                </p>
              </div>
            </div>
          </Card>
        </section>
      </section>
    </main>
  );
}
