import React, { useState } from 'react';
import './index.css';
import { HistoricalAnalytics } from './components/HistoricalAnalytics.jsx';
import { MaterialPredictor } from './components/MaterialPredictor.jsx';
import { ChatBot } from './components/ChatBot.jsx';
import logo from '../Assets/g.png';
import thumpVideo from '../Assets/Gorilla_Chest_Thumping_Animation_Generated.mp4';

function App() {
  const [activeTab, setActiveTab] = useState('waste');
  const [showChat, setShowChat] = useState(false);
  const [showVideo, setShowVideo] = useState(false);

  const handleGorillaClick = () => {
    setShowVideo(true);
    setShowChat(false);
  };

  return (
    <>
      <div className="dashboard-header">
        <div className="header-title-container">
          <img 
            src={logo} 
            alt="Gorilla Semiconductors Logo" 
            className="gorilla-logo" 
            onClick={handleGorillaClick}
          />
          <h1 className="header-title">Gorilla Semiconductors</h1>
        </div>
      </div>

      <div className="tabs-container">
        <button 
          className={`tab-btn ${activeTab === 'waste' ? 'active' : ''}`}
          onClick={() => setActiveTab('waste')}
        >
          Historical Waste Analysis
        </button>
        <button 
          className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          Material Prediction
        </button>
      </div>

      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'waste' ? <HistoricalAnalytics /> : <MaterialPredictor />}
      </div>

      <ChatBot isOpen={showChat} onClose={() => setShowChat(false)} />

      {showVideo && (
        <div className="thump-video-overlay" onClick={() => { setShowVideo(false); setShowChat(true); }}>
          <video 
            src={thumpVideo} 
            autoPlay 
            className="thump-video"
            onEnded={() => { setShowVideo(false); setShowChat(true); }}
          />
        </div>
      )}
    </>
  );
}

export default App;
