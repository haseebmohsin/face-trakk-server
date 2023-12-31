const mongoose = require('mongoose');

// Define the Cluster schema
const clusterSchema = new mongoose.Schema(
  {
    isActive: { type: Boolean, default: true },

    faceImagesArray: [
      {
        faceName: { type: String, required: true },
        faceImage: { type: Buffer, required: true },
      },
    ],
  },
  {
    timestamps: true,
  }
);

// Export the Cluster schema as a Mongoose model
module.exports = mongoose.model('Cluster', clusterSchema);
