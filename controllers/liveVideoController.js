const asyncHandler = require('express-async-handler');
const { spawn } = require('child_process');
const fs = require('fs');
const Thumbnail = require('../models/thumbnailModel');
const { executePythonScript } = require('../utils/pythonUtils');

/**
 * @desc    Detect People in a video and abstract data from it.
 * @route   POST /api/liveVideo
 * @access  Private
 */
const liveVideo = asyncHandler(async (req, res) => {
  const pythonScriptPath = 'scripts/test.py';

  await executePythonScript(pythonScriptPath);
  console.log('Script execution completed successfully');

  setTimeout(() => {
    // Read data.json file which is created by running the script.
    fs.readFile('scripts/data.json', 'utf8', async (err, data) => {
      if (err) {
        res.status(400);
        throw new Error('Error reading data');
      }

      const thumbnails = JSON.parse(JSON.parse(data));

      // Convert the thumbnail data to a Buffer before saving
      const thumbnailData = thumbnails.map((thumbnail) => ({
        ...thumbnail,
        // Convert the thumbnail from base64 to Buffer
        thumbnail: Buffer.from(thumbnail.thumbnail, 'base64'),
      }));

      // Delete all previous records from the database
      await Thumbnail.deleteMany({});

      // Save the new thumbnail data to the database
      await Thumbnail.insertMany(thumbnailData);

      const thumbnailsDataFromDatabase = await Thumbnail.find();

      // Convert thumbnails to an array of objects with base64 encoded thumbnails
      const thumbnailsData = thumbnailsDataFromDatabase.map((thumbnail) => ({
        ...thumbnail._doc,
        thumbnail: thumbnail.thumbnail.toString('base64'),
      }));

      res.status(200).json({ message: 'Successfully data received', thumbnails: thumbnailsData });
    });
  }, 2000);
});

/**
 * @desc    Get all thumbnail data
 * @route   GET /api/thumbnails
 * @access  Private
 */
const getThumbnails = asyncHandler(async (req, res) => {
  const thumbnails = await Thumbnail.find();

  // Convert thumbnails to an array of objects with base64 encoded thumbnails
  const thumbnailsData = thumbnails.map((thumbnail) => ({
    ...thumbnail._doc,
    thumbnail: thumbnail.thumbnail.toString('base64'),
  }));

  res.status(200).json({ message: 'Successfully data received', thumbnails: thumbnailsData });
});

module.exports = {
  liveVideo,
  getThumbnails,
};
