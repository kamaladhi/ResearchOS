// API Configuration
const API_BASE_URL = "http://127.0.0.1:8000";

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

const ingestInput = document.getElementById('ingest-input');
const ingestBtn = document.getElementById('ingest-btn');
const ingestStatus = document.getElementById('ingest-status');
const ingestText = document.getElementById('ingest-text');

// --- Ingestion Logic ---
ingestBtn.addEventListener('click', async () => {
    const query = ingestInput.value.trim();
    if (!query) return;

    // Show Loading State
    ingestBtn.disabled = true;
    ingestStatus.classList.remove('hidden');
    ingestText.innerText = "Downloading paper & extracting entities...";

    try {
        const response = await fetch(`${API_BASE_URL}/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, max_results: 1 })
        });

        const data = await response.json();
        
        if (response.ok) {
            ingestText.innerText = "✅ Successfully Ingested!";
            addAiMessage(`I have successfully ingested the paper: **${data.details[0].title}** into the Knowledge Graph. You can now ask me questions about it!`);
        } else {
            ingestText.innerText = "❌ Ingestion Failed.";
            console.error(data);
        }
    } catch (error) {
        ingestText.innerText = "❌ Network Error.";
        console.error(error);
    } finally {
        ingestBtn.disabled = false;
        setTimeout(() => {
            if (ingestText.innerText.includes("✅")) {
                ingestStatus.classList.add('hidden');
                ingestInput.value = '';
            }
        }, 3000);
    }
});

// --- Chat Logic ---
function addUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.innerHTML = `
        <div class="avatar user-avatar">👤</div>
        <div class="bubble">${text}</div>
    `;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function addAiMessage(data) {
    let answerText = "";
    let citations = [];
    let followUps = [];
    
    // Check if we passed a string or a structured object
    if (typeof data === 'string') {
        answerText = data;
    } else {
        answerText = data.answer;
        citations = data.citations || [];
        followUps = data.follow_up_questions || [];
    }

    // Basic Markdown bold parsing for aesthetic
    const formattedText = answerText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Build Citations HTML
    let citationsHtml = '';
    if (citations.length > 0) {
        citationsHtml = '<div class="citations-box"><div class="citations-title">Sources:</div><ul>';
        citations.forEach(c => citationsHtml += `<li>${c}</li>`);
        citationsHtml += '</ul></div>';
    }

    // Build Follow-ups HTML
    let followUpsHtml = '';
    if (followUps.length > 0) {
        followUpsHtml = '<div class="followups-box">';
        followUps.forEach(q => {
            followUpsHtml += `<button class="followup-chip" onclick="triggerFollowUp('${q.replace(/'/g, "\\'")}')">${q}</button>`;
        });
        followUpsHtml += '</div>';
    }

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai-message';
    msgDiv.innerHTML = `
        <div class="avatar ai-avatar">🤖</div>
        <div class="bubble">
            ${formattedText}
            ${citationsHtml}
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    
    if (followUpsHtml) {
        const chipDiv = document.createElement('div');
        chipDiv.className = 'message ai-message follow-up-container';
        chipDiv.innerHTML = `<div class="avatar ai-avatar" style="visibility:hidden;"></div>${followUpsHtml}`;
        chatHistory.appendChild(chipDiv);
    }
    
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Global function for the follow-up chips to trigger a new message
window.triggerFollowUp = function(question) {
    chatInput.value = question;
    handleChat();
}

function addTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai-message typing-msg';
    msgDiv.id = 'typing-indicator';
    msgDiv.innerHTML = `
        <div class="avatar ai-avatar">🤖</div>
        <div class="bubble typing-indicator">
            <span></span><span></span><span></span>
        </div>
    `;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function handleChat() {
    const question = chatInput.value.trim();
    if (!question) return;

    // UI Updates
    chatInput.value = '';
    addUserMessage(question);
    addTypingIndicator();

    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        removeTypingIndicator();

        if (response.ok) {
            addAiMessage(data); // Pass the whole data object now!
        } else {
            addAiMessage("Sorry, I encountered an error. Is the FastAPI server running?");
        }
    } catch (error) {
        removeTypingIndicator();
        addAiMessage("Network Error: Could not reach the API. Make sure `python src/api.py` is running on port 8000!");
    }
}

// Event Listeners for Chat
sendBtn.addEventListener('click', handleChat);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleChat();
});
