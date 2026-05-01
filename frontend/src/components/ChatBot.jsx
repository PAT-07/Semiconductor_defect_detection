import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export const ChatBot = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Dragging state
  const [position, setPosition] = useState({ 
    x: typeof window !== 'undefined' ? window.innerWidth - 400 : 0, 
    y: typeof window !== 'undefined' ? window.innerHeight - 650 : 0 
  });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef({ startX: 0, startY: 0, initialX: 0, initialY: 0 });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle Dragging
  const handlePointerDown = (e) => {
    // Don't drag if clicking the close button
    if (e.target.tagName.toLowerCase() === 'button') return;
    setIsDragging(true);
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      initialX: position.x,
      initialY: position.y
    };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setPosition({
      x: dragRef.current.initialX + dx,
      y: dragRef.current.initialY + dy
    });
  };

  const handlePointerUp = (e) => {
    setIsDragging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        messages: [...messages, userMessage].map(m => ({ role: m.role, content: m.content }))
      });
      
      if (response.data.response) {
        setMessages(prev => [...prev, { role: 'model', content: response.data.response }]);
      } else if (response.data.error) {
        setMessages(prev => [...prev, { role: 'model', content: `Error: ${response.data.error}` }]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, { role: 'model', content: "GRRR... I couldn't reach the server. Is it running?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="chat-widget-container" 
      style={{ left: position.x, top: position.y, margin: 0 }}
    >
      <div 
        className="chat-header" 
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <h3>🦍 Gorilla Bot</h3>
        <button className="chat-close-btn" onClick={onClose}>×</button>
      </div>
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-message ${msg.role}`}>
            <div className="chat-bubble">
              {msg.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="chat-message model">
            <div className="chat-bubble loading">
              Thinking... 🍌
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input-area" onSubmit={handleSend}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about defects, KPIs, or forecasts..." 
          className="chat-input"
        />
        <button type="submit" className="chat-send-btn" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};
