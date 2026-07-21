// Thin fetch wrapper. All paths are relative /api/* so the same build works in
// dev (Vite proxy) and prod (nginx proxy).

async function req(path, options) {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const api = {
  health: () => req("/health"),
  portfolio: () => req("/portfolio"),
  accounts: (riskTier) =>
    req(`/accounts${riskTier ? `?risk_tier=${encodeURIComponent(riskTier)}` : ""}`),
  account: (id) => req(`/accounts/${id}`),
  propose: (id) => req(`/accounts/${id}/propose`, { method: "POST" }),
  decide: (proposalId, decision, note) =>
    req(`/proposals/${proposalId}/decision`, {
      method: "POST",
      body: JSON.stringify({ decision, note }),
    }),
  memory: () => req("/memory"),
  churnMetrics: () => req("/metrics/churn"),
  intentMetrics: () => req("/metrics/intents"),
  causalMetrics: () => req("/metrics/causal"),
  llmopsMetrics: () => req("/metrics/llmops"),
  plotUrl: (name) => `/api/plots/${name}`,
};
