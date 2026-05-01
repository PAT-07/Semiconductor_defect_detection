import React from 'react';
import '../index.css';

const KPICard = ({ title, value, subtitle, color }) => {
  return (
    <div className="kpi-card">
      <p className="kpi-title">{title}</p>
      <h2 className="kpi-value" style={{ color: color }}>{value}</h2>
      {subtitle && <p className="kpi-subtitle">{subtitle}</p>}
    </div>
  );
};

export default KPICard;
