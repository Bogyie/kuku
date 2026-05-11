import type { AiConfig, AiProvider } from "./types";

const AI_CHAT_SETTINGS_PLUGIN_ID = "ai-chat";
const AI_CHAT_SECURE_KEYS = ["apiKey"] as const;
const DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview";
const DEFAULT_REMOTE_MODEL = DEFAULT_GEMINI_MODEL;
const DEFAULT_CODEX_MODEL = "gpt-5.5";
const DEFAULT_MODEL = DEFAULT_REMOTE_MODEL;
const DEFAULT_PROVIDER: AiProvider = "remote";
const DEFAULT_SERVER_URL =
  import.meta.env.VITE_KUKU_API_URL?.trim() ||
  (import.meta.env.PROD ? "https://api.kuku.mom" : "http://localhost:8080");
// Internal guardrails: these are intentionally kept out of the settings UI.
const DEFAULT_ROUND_LIMIT = 12;
const DEFAULT_PROXY_TIMEOUT_MS = 15_000;

function defaultModelForProvider(provider: AiProvider): string {
  if (provider === "codex") return DEFAULT_CODEX_MODEL;
  if (provider === "gemini") return DEFAULT_GEMINI_MODEL;
  return DEFAULT_REMOTE_MODEL;
}

function aiProviderFrom(value: unknown): AiProvider | null {
  if (value === "gemini" || value === "remote" || value === "codex") return value;
  return null;
}

function pinnedModelForConfig(provider: AiProvider, model: string | null): string {
  if (provider === "codex" && typeof model === "string" && model.trim().length > 0) {
    return model.trim();
  }
  return defaultModelForProvider(provider);
}

function createDefaultAiConfig(): AiConfig {
  return {
    provider: DEFAULT_PROVIDER,
    apiKey: null,
    model: defaultModelForProvider(DEFAULT_PROVIDER),
    serverUrl: DEFAULT_SERVER_URL,
    roundLimit: DEFAULT_ROUND_LIMIT,
    proxyToolTimeoutMs: DEFAULT_PROXY_TIMEOUT_MS,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeAiConfig(raw: unknown): AiConfig {
  const defaults = createDefaultAiConfig();
  if (!isRecord(raw)) return defaults;
  const provider = aiProviderFrom(raw.provider) ?? DEFAULT_PROVIDER;

  return {
    provider,
    apiKey: typeof raw.apiKey === "string" && raw.apiKey.trim().length > 0 ? raw.apiKey : null,
    model: pinnedModelForConfig(provider, typeof raw.model === "string" ? raw.model : null),
    serverUrl:
      typeof raw.serverUrl === "string" && raw.serverUrl.trim().length > 0
        ? raw.serverUrl
        : defaults.serverUrl,
    roundLimit:
      typeof raw.roundLimit === "number" && Number.isFinite(raw.roundLimit) && raw.roundLimit > 0
        ? raw.roundLimit
        : defaults.roundLimit,
    proxyToolTimeoutMs:
      typeof raw.proxyToolTimeoutMs === "number" &&
      Number.isFinite(raw.proxyToolTimeoutMs) &&
      raw.proxyToolTimeoutMs > 0
        ? raw.proxyToolTimeoutMs
        : defaults.proxyToolTimeoutMs,
  };
}

export {
  AI_CHAT_SETTINGS_PLUGIN_ID,
  AI_CHAT_SECURE_KEYS,
  DEFAULT_MODEL,
  DEFAULT_PROVIDER,
  DEFAULT_PROXY_TIMEOUT_MS,
  DEFAULT_ROUND_LIMIT,
  DEFAULT_SERVER_URL,
  createDefaultAiConfig,
  aiProviderFrom,
  defaultModelForProvider,
  normalizeAiConfig,
  pinnedModelForConfig,
};
