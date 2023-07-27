const mongoose = require('mongoose');

// Define the schema
const thumbnailSchema = new mongoose.Schema({
  name: {
    type: String,
  },
  startTime: {
    type: String,
  },
  endTime: {
    type: String,
  },
  coverageTime: {
    type: Number,
  },
  thumbnail: {
    type: Buffer,
  },
});

// Create a model from the schema
const Thumbnail = mongoose.model('Thumbnail', thumbnailSchema);

module.exports = Thumbnail;
