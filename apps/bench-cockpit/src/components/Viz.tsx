import React from 'react';
import Scatter from './Scatter';
import Heatmap from './Heatmap';
import FailMatrix from './FailMatrix';
import { Cell } from '../types';

interface Props {
  cells: Cell[];
  onSelect?: (c: Cell) => void;
}

export default function Viz({ cells, onSelect }: Props) {
  return (
    <div className="view-stack" data-testid="viz-view">
      <Scatter cells={cells} onSelect={onSelect} />
      <Heatmap cells={cells} />
      <FailMatrix cells={cells} onSelect={onSelect} />
    </div>
  );
}
