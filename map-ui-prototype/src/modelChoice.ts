import type { ModelCatalogue } from './agentClient';

type Group = ModelCatalogue['providers'][number];

/** The suffix a provider's group label gets when the server left models out of it. */
export function hiddenNote(hidden: Group['hidden']): string {
  const n = Object.keys(hidden || {}).length;
  return n ? ` — ${n} hidden: no tool calls` : '';
}

/**
 * Why the saved model can no longer be used, or null when it still can (or when that cannot be
 * told). The choice lives in localStorage, and a `<select>` whose value matches no option SHOWS
 * its first enabled one, "Agent default", while every turn still sends the saved id. A model the
 * catalogue stops offering would then fail every turn behind a picker that looks fine.
 *
 * Unknowable is not unavailable: no catalogue, a provider whose list could not be fetched
 * (`stale`), or one that is not configured, so its list is the fallback ids. Those keep the
 * choice.
 */
export function unavailableChoice(model: string | undefined, provider: string | undefined,
                                  catalogue: ModelCatalogue | null): string | null {
  if (!model || !catalogue) return null;
  const groups = catalogue.providers || [];
  if (groups.some((g) => g.models.includes(model))) return null;
  const owner = groups.find((g) => g.provider === provider)
    || groups.find((g) => g.hidden && model in g.hidden);
  const why = owner?.hidden?.[model];
  if (why) return `${model} is no longer offered: ${why}.`;
  if (owner) {
    return owner.stale || !owner.configured ? null
      : `${model} is no longer offered by ${owner.label}.`;
  }
  // No provider recorded with the choice (saved before the picker stored one), and none lists
  // it. Only conclusive when every list is a live one.
  return groups.some((g) => g.stale || !g.configured) ? null : `${model} is no longer offered.`;
}
