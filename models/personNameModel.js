const mongoose = require('mongoose');

// Define the PersonName schema
const personNameSchema = new mongoose.Schema(
  {
    personName: { type: String, required: true },
  },
  {
    timestamps: true,
  }
);

// Export the Person schema as a Mongoose model
module.exports = mongoose.model('Person', personNameSchema);
