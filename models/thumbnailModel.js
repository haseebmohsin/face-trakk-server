const mongoose = require('mongoose');

// Define the schema
const thumbnailSchema = new mongoose.Schema(
  {
    thumbnail: Buffer,
    name: String,
    timestamps: [String],
    startTime: [String],
    endTime: [String],
    coverageTime: String,
  },
  {
    timestamps: true,
  }
);

// Create a model from the schema
const Thumbnail = mongoose.model('Thumbnail', thumbnailSchema);

module.exports = Thumbnail;
