const WebSocket = require('ws');

// Create WebSocket server
const wss = new WebSocket.Server({ noServer: true });

// Handle WebSocket connections
wss.on('connection', (ws) => {
  console.log('WebSocket connection established');

  // Handle messages received from the client
  ws.on('message', (message) => {
    // Handle incoming messages if required
  });

  // Handle WebSocket connection close
  ws.on('close', () => {
    console.log('WebSocket connection closed');
  });
});

module.exports = wss;
