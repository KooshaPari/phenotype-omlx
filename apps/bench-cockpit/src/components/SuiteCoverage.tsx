import type { SuiteCoverageRow } from '../types';

type Props = {
  rows: SuiteCoverageRow[];
  onJumpToSuite?: (suite: string) => void;
};

function armsLabel(row: SuiteCoverageRow): string {
  const parts: string[] = [];
  if (row.has_stock) parts.push('stock');
  if (row.has_ours) parts.push('ours');
  for (const a of row.experiment_arms ?? []) parts.push(a);
  return parts.length ? parts.join(' · ') : '—';
}

export function SuiteCoverage({ rows, onJumpToSuite }: Props) {
  if (!rows.length) return null;
  const present = rows.filter((r) => r.present).length;
  const gaps = rows.filter((r) => !r.present);
  const stockOurs = rows.filter((r) => r.has_stock && r.has_ours).length;

  return (
    <div className="ds" style={{ marginTop: 20 }} data-testid="suite-coverage">
      <h3 className="section-title">Suite coverage</h3>
      <p className="muted" style={{ marginTop: 4 }}>
        Catalog {rows.length} · loaded {present} · stock+ours paired {stockOurs} · gaps {gaps.length}.
        V5 ablation is stock/ours; extended matrices land as experiment arms (e.g. minimax-m3).
      </p>
      <div className="table-wrap" style={{ marginTop: 12, overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Suite</th>
              <th>Status</th>
              <th>Arms</th>
              <th>Cells</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.suite}
                className={r.present ? undefined : 'faint'}
                style={{ cursor: r.present && onJumpToSuite ? 'pointer' : undefined }}
                onClick={() => {
                  if (r.present && onJumpToSuite) onJumpToSuite(r.suite);
                }}
              >
                <td className="mono">{r.suite}</td>
                <td>{r.present ? (r.has_stock && r.has_ours ? 'paired' : 'partial') : 'missing'}</td>
                <td className="mono faint">{armsLabel(r)}</td>
                <td>{r.n_cells}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {gaps.length > 0 && (
        <p className="faint mono" style={{ marginTop: 8 }}>
          Missing: {gaps.map((g) => g.suite).join(', ')}
        </p>
      )}
    </div>
  );
}
