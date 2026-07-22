/** Ablation peers in V5 stock-vs-ours runs. */
export const ABLATION_VARIANTS = new Set(['stock', 'ours']);

/**
 * Non-ablation cell variants are auxiliary roles (judge / evaluator / distiller),
 * not peer models to compare against stock/ours.
 */
export function isAblationVariant(variant: string | undefined | null): boolean {
  return Boolean(variant && ABLATION_VARIANTS.has(variant));
}

export function isAuxRole(variant: string | undefined | null): boolean {
  return Boolean(variant) && !isAblationVariant(variant);
}

/** Human label for an aux arm slug (e.g. minimax-m3 → judge/eval). */
export function auxRoleLabel(arm: string): string {
  const a = arm.toLowerCase();
  if (a.includes('minimax') || a.includes('judge') || a.includes('eval')) {
    return `${arm} (judge / eval)`;
  }
  if (a.includes('distill')) {
    return `${arm} (distiller)`;
  }
  return `${arm} (aux)`;
}

export function ablationVariants(variants: Iterable<string>): string[] {
  return [...variants].filter(isAblationVariant).sort();
}

export function auxVariants(variants: Iterable<string>): string[] {
  return [...variants].filter(isAuxRole).sort();
}

/** Sidebar model pill: never join aux arms into the peer model string. */
export function displayModelLabel(
  metaModel: string | undefined,
  variants: string[] | undefined,
): { model: string; aux: string[] } {
  const all = variants?.length ? variants : (metaModel ? metaModel.split('+').map((s) => s.trim()) : []);
  const ablation = ablationVariants(all);
  const aux = auxVariants(all);
  let model = metaModel || '—';
  if (aux.length && model.includes('+')) {
    const parts = model.split('+').map((s) => s.trim()).filter(isAblationVariant);
    if (parts.length) model = parts.join('+');
  }
  if ((!metaModel || isAuxRole(metaModel) || aux.some((a) => model.includes(a))) && ablation.length) {
    model = ablation.join('+');
  }
  return { model: model || '—', aux };
}
