import type { SuiteCoverageRow } from '../types';
import { auxRoleLabel } from '../lib/arms';

type Props = {
  rows: SuiteCoverageRow[];
  onJumpToSuite?: (suite: string) => void;
};

function peersLabel(row: SuiteCoverageRow): string {
  const parts: string[] = [];
  if (row.has_stock) parts.push('stock');
  if (row.has_ours) parts.push('ours');
  return parts.length ? parts.join(' · ') : '—';
}

function auxLabel(row: SuiteCoverageRow): string {
  const arms = row.experiment_arms ?? [];
  if (!arms.length) return '—';
  return arms.map(auxRoleLabel).join(' · ');
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
        Peers are stock/ours only. Aux columns are judge / evaluator / distiller matrices — not competing models.
      </p>
      <div className="table-wrap" style={{ marginTop: 12, overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Suite</th>
              <th>Status</th>
              <th>Peers</th>
              <th>Aux (judge/eval)</th>
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
                <td className="mono faint">{peersLabel(r)}</td>
                <td className="mono faint">{auxLabel(r)}</td>
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
