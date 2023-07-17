const mongoose = require('mongoose');

// Define the Cluster schema
const clusterSchema = new mongoose.Schema(
  {
    clusterName: { type: String, required: true },

    faceImages: [
      {
        imageName: { type: String, required: true },
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
