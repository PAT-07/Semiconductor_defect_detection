import React, { useState, useEffect } from 'react';
import axios from 'axios';
import KPICard from './KPICard';

export const MaterialPredictor = () => {
    const [scans, setScans] = useState(1300);
    const [failRate, setFailRate] = useState(97);
    const [prediction, setPrediction] = useState(null);
    const [modelStatus, setModelStatus] = useState(null);

    useEffect(() => {
        axios.get('http://127.0.0.1:8000/api/model/status').then(res => {
            setModelStatus(res.data);
        }).catch(() => setModelStatus({loaded: false}));
    }, []);

    const handlePredict = async () => {
        try {
            const res = await axios.post('http://127.0.0.1:8000/api/predict', { scans, fail_rate: failRate });
            setPrediction(res.data);
        } catch(e) {
            console.error(e);
        }
    };

    if (!modelStatus) return <div>Loading...</div>;

    return (
        <div>
            <div className="glass-card predictor-header">
                <div>
                    <h3 className="predictor-title">Prediction Model</h3>
                    <p style={{margin:0, color: modelStatus.loaded ? 'var(--accent3)' : 'var(--danger)'}}>
                        {modelStatus.loaded ? 'Model loaded' : 'No model found'}
                    </p>
                </div>
                {modelStatus.loaded && (
                    <p style={{color: 'var(--text-muted)', fontFamily:'monospace'}}>
                        R² = {modelStatus.metrics.r2} | MAE = {modelStatus.metrics.mae}%
                    </p>
                )}
            </div>

            <div className="glass-card">
                <h3 className="predictor-title" style={{marginBottom: '20px'}}>Forecast Parameters</h3>
                <div className="predictor-form">
                    <div className="slider-container">
                        <label>Expected Daily Production ({scans} wafers)</label>
                        <input type="range" min="100" max="2000" step="50" value={scans} onChange={(e) => setScans(parseInt(e.target.value))} />
                    </div>
                    <div className="slider-container">
                        <label>Expected Defect Rate ({failRate}%)</label>
                        <input type="range" min="0" max="100" step="5" value={failRate} onChange={(e) => setFailRate(parseInt(e.target.value))} />
                    </div>
                </div>
                <button className="btn-primary" onClick={handlePredict}>Predict Material Needs</button>
            </div>

            {prediction && (
                <div className="glass-card prediction-result">
                    <div className="kpi-container" style={{marginBottom: 0}}>
                        <KPICard title="Daily Production" value={prediction.total_scans} subtitle="wafers" color="var(--accent)" />
                        <KPICard title="Expected Defect Rate" value={`${prediction.fail_rate}%`} subtitle={`~${Math.round((prediction.total_scans*prediction.fail_rate)/100)} defective`} color="var(--danger)" />
                        <KPICard title="Avg Waste/Wafer" value={`${prediction.avg_waste_per_wafer}%`} subtitle="per defective wafer loss" color="var(--warning)" />
                        <KPICard title="Total Daily Waste" value={`${prediction.total_daily_waste} wafers`} subtitle="total estimated loss" color="var(--danger)" />
                    </div>
                </div>
            )}
        </div>
    );
};
