const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const Face = require('../models/faceModel');
const fs = require('fs');
const asyncHandler = require('express-async-handler');

// Create a storage engine for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/videos/'); // Specify the destination folder
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const extname = path.extname(file.originalname);
    cb(null, 'video-' + uniqueSuffix + extname);
  },
});

// File upload middleware
const upload = multer({ storage });

/**
 * @desc    Upload Video for face recognition
 * @route   POST /api/videoUpload
 * @access  Private
 */
const videoUpload = asyncHandler(async (req, res) => {
  upload.single('video')(req, res, async (err) => {
    if (err) {
      // Handle any error that occurs during the upload
      console.error('Error uploading file:', err);
      return res.status(500).json({ error: 'Error uploading file' });
    }

    // File upload is successful
    const videoPath = `${req.file.path}`;

    // Delete all data from Face schema after video upload
    await Face.deleteMany({});

    await executePythonScript(videoPath);

    console.log('Script execution completed successfully');

    // wait for the faces to be written in the folder and then upload to the database
    await postImagesToDatabase('scripts/faces');

    setTimeout(async () => {
      // Fetch faces from the database
      const faces = await Face.find();

      res.status(200).json({ message: 'Video uploaded and script executed successfully', faces });
    }, 3000);
  });
});

/**
 * @desc    get faces data
 * @route   GET /api/faces
 * @access  Private
 */
// const getFaces = async (req, res) => {
//   // Fetch faces from the database
//   const images = await Face.find();

//   res.status(200).json({ message: 'Faces retrieved', images });
// };

const postImagesToDatabase = (folderPath) => {
  fs.readdir(folderPath, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return;
    }

    files.forEach((file) => {
      const imagePath = `${folderPath}/${file}`;
      const path = require('path');

      // Create a new image document
      const image = new Face({
        name: path.parse(file).name,
        path: file,
      });

      // Save the image document to the database
      image.save((error) => {
        if (error) {
          console.error(`Error saving image ${file} to database:`, error);
        } else {
          console.log(`Image ${file} uploaded and saved to the database`);
        }
      });
    });
  });
};

const executePythonScript = (videoPath, frameNumber = 0) => {
  console.log('Script start Running');

  const pythonExecutablePath = 'D:/ProgramData/anaconda3/envs/tf/python.exe';
  const pythonScriptPath = '../scripts/test.py';

  return new Promise((resolve, reject) => {
    const process = spawn(pythonExecutablePath, [pythonScriptPath, videoPath, frameNumber], {
      cwd: path.dirname(pythonScriptPath),
    });

    process.on('close', (code) => {
      if (code === 0) {
        // Script execution successful
        resolve();
      } else {
        // Script execution failed
        console.log('Script stop Running');
        console.log('Script execution failed for some reason');

        reject(new Error('Script execution failed'));
      }
    });
  });
};

module.exports = {
  videoUpload,
};
