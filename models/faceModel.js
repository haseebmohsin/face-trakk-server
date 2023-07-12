// models/faceModel.js
const mongoose = require('mongoose');

const faceSchema = new mongoose.Schema(
  {
    name: String,
    path: String,
  },
  { timestamps: true }
);

const Face = mongoose.model('Face', faceSchema);

module.exports = Face;
