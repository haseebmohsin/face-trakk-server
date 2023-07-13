require('dotenv').config();
const express = require('express');
const cors = require('cors');
const app = express();
require('dotenv').config();
const apiRoutes = require('./routes/api');
const connectDB = require('./config/db');
const errorHandler = require('./middleware/errorMiddleware');

// Middleware
app.use(cors());
app.use(express.json());

// Database connection
connectDB();

// Routes
app.use('/api', apiRoutes);

// Error handling middleware
app.use(errorHandler);

// Start the server
const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
