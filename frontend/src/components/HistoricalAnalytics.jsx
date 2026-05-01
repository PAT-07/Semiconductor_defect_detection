import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, BarElement } from 'chart.js';
import { Pie, Bar, Line } from 'react-chartjs-2';
import KPICard from './KPICard';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, BarElement);

const COLORS = {
  accent: '#f472b6',
  accent2: '#38bdf8',
  accent3: '#4ade80',
  danger: '#fb7185',
  warning: '#fbbf24',
  text: '#000000',
  textMuted: '#3f3f46'
};

const DEFECT_COLORS = {
  'Center': '#ef4444', 'Donut': '#f59e0b', 'Edge-Loc': '#10b981',
  'Edge-Ring': '#3b82f6', 'Loc': '#8b5cf6', 'Random': '#ec4899',
  'Scratch': '#06b6d4', 'Near-full': '#f97316', 'None': '#6b7280',
  'Undetected': '#374151'
};

const chartOptions = {
    color: COLORS.text,
    plugins: {
        legend: {
            labels: { color: COLORS.textMuted }
        }
    },
    scales: {
        x: { ticks: { color: COLORS.textMuted }, grid: { color: 'rgba(0,0,0,0.1)' } },
        y: { ticks: { color: COLORS.textMuted }, grid: { color: 'rgba(0,0,0,0.1)' } }
    }
};
const pieOptions = {
    color: COLORS.text,
    plugins: { legend: { labels: { color: COLORS.textMuted } } }
};

export const HistoricalAnalytics = () => {
    const [kpis, setKpis] = useState(null);
    const [defects, setDefects] = useState(null);
    const [waste, setWaste] = useState(null);
    const [trends, setTrends] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            const [kRes, dRes, wRes, tRes] = await Promise.all([
                axios.get('http://127.0.0.1:8000/api/kpi'),
                axios.get('http://127.0.0.1:8000/api/charts/defects'),
                axios.get('http://127.0.0.1:8000/api/charts/waste'),
                axios.get('http://127.0.0.1:8000/api/charts/trends')
            ]);
            setKpis(kRes.data);
            setDefects(dRes.data);
            setWaste(wRes.data);
            setTrends(tRes.data);
        };
        fetchData();
    }, []);

    if (!kpis || !defects || !waste || !trends) return <div>Loading Analytics...</div>;

    const pieData = {
        labels: defects.predictions.map(d => d.defect_type),
        datasets: [{
            data: defects.predictions.map(d => d.count),
            backgroundColor: defects.predictions.map(d => DEFECT_COLORS[d.defect_type] || COLORS.textMuted),
            borderColor: 'transparent'
        }]
    };

    const barData = {
        labels: waste.waste_by_type.map(w => w.defect_type),
        datasets: [{
            label: 'Total Material Waste (Wafers)',
            data: waste.waste_by_type.map(w => w.total_waste),
            backgroundColor: waste.waste_by_type.map(w => DEFECT_COLORS[w.defect_type] || COLORS.textMuted)
        }]
    };

    const trendData = {
        labels: trends.dates,
        datasets: [{
            label: 'Fail Rate %',
            data: trends.fail_rate,
            borderColor: COLORS.danger,
            backgroundColor: 'rgba(251, 113, 133, 0.4)',
            fill: true,
            yAxisID: 'y'
        }]
    };

    const wasteTrendData = {
        labels: trends.dates,
        datasets: [{
            label: 'Total Lost Wafers',
            data: trends.waste,
            borderColor: COLORS.warning,
            backgroundColor: 'rgba(251, 191, 36, 0.4)',
            fill: true,
            yAxisID: 'y'
        }]
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div className="kpi-container">
                <KPICard title="Total Scans" value={kpis.total_scans.toLocaleString()} subtitle="wafers inspected" color={COLORS.accent} />
                <KPICard title="Pass Rate" value={`${kpis.pass_rate}%`} subtitle={`${kpis.pass_count.toLocaleString()} passed`} color={COLORS.accent3} />
                <KPICard title="Fail Rate" value={`${kpis.fail_rate}%`} subtitle={`${kpis.fail_count.toLocaleString()} defective`} color={COLORS.danger} />
                <KPICard title="Scrapped" value={kpis.scrap_count.toLocaleString()} subtitle="routed to scrap" color={COLORS.warning} />
                <KPICard title="Avg Waste/Wafer" value={`${kpis.avg_waste}%`} subtitle="per defective wafer" color={COLORS.danger} />
                <KPICard title="Avg Confidence" value={kpis.avg_confidence} subtitle="model certainty" color={COLORS.accent3} />
            </div>

            <div className="charts-master-grid">
                <div className="glass-card chart-card">
                    <h3 style={{marginTop:0, marginBottom: '8px', fontSize: '14px'}}>YOLOv8 Predicted Distributions</h3>
                    <div className="canvas-container">
                        <Pie data={pieData} options={{...pieOptions, maintainAspectRatio: false}} />
                    </div>
                </div>
                <div className="glass-card chart-card">
                    <h3 style={{marginTop:0, marginBottom: '8px', fontSize: '14px'}}>Total Material Waste by Predict Defect</h3>
                    <div className="canvas-container">
                        <Bar data={barData} options={{...chartOptions, maintainAspectRatio: false}} />
                    </div>
                </div>
                <div className="glass-card chart-card">
                    <h3 style={{marginTop:0, marginBottom: '8px', fontSize: '14px'}}>Daily Defect Rate Over Time</h3>
                    <div className="canvas-container">
                        <Line data={trendData} options={{...chartOptions, maintainAspectRatio: false}} />
                    </div>
                </div>
                <div className="glass-card chart-card">
                    <h3 style={{marginTop:0, marginBottom: '8px', fontSize: '14px'}}>Daily Material Waste Over Time</h3>
                    <div className="canvas-container">
                        <Line data={wasteTrendData} options={{...chartOptions, maintainAspectRatio: false}} />
                    </div>
                </div>
            </div>
        </div>
    );
};
