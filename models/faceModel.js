const mongoose = require('mongoose');

const faceSchema = new mongoose.Schema(
  {
    name: String,
    image: { type: Buffer },
  },
  { timestamps: true }
);

const Face = mongoose.model('Face', faceSchema);

module.exports = Face;
