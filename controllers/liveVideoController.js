const asyncHandler = require('express-async-handler');
const { spawn } = require('child_process');
const fs = require('fs');
const Thumbnail = require('../models/thumbnailModel');

/**
 * @desc    Detect People in a video and abstract data from it.
 * @route   POST /api/liveVideo
 * @access  Private
 */
const liveVideo = asyncHandler(async (req, res) => {
  const pythonScriptPath = 'scripts/fr_model/detect.py';

  await executePythonScript(pythonScriptPath);
  console.log('Script execution completed successfully');

  setTimeout(() => {
    // Read data.json file
    fs.readFile('scripts/fr_model/data.json', 'utf8', async (err, data) => {
      if (err) {
        res.status(400);
        throw new Error('Error reading data.json');
      }

      const thumbnails = JSON.parse(JSON.parse(data));

      // Convert the thumbnail data to a Buffer before saving
      const thumbnailData = thumbnails.map((thumbnail) => ({
        ...thumbnail,
        // Convert the thumbnail data from base64 to Buffer
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

/**
 * @desc    Execute Python script
 * @param   {string} pythonScriptPath - Path of the Python script file
 * @param   {Object} args - Arguments to be passed to the `spawn` function
 * @returns {Promise<void>}
 */
const executePythonScript = (pythonScriptPath, args = []) => {
  console.log('Script start Running');

  const pythonExecutablePath = `${process.env.PYTHON_EXE_PATH_TF_TORCH}`;

  return new Promise((resolve, reject) => {
    const process = spawn(pythonExecutablePath, [pythonScriptPath, ...args]);

    process.stdout.on('data', (data) => {
      console.log(`Python script output: ${data}`);
    });

    process.on('close', (code) => {
      if (code === 0) {
        // Script execution successful
        resolve();
      } else {
        // Script execution failed
        console.log('Script execution failed for some reason');

        reject(new Error('Script execution failed'));
      }
    });
  });
};

module.exports = {
  liveVideo,
  getThumbnails,
};
