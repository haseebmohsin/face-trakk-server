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
const videoUpload = async (req, res) => {
  try {
    upload.single('video')(req, res, async (err) => {
      if (err) {
        console.error('Error uploading file:', err);
        return res.status(500).json({ error: 'Error uploading file' });
      }

      const videoPath = `${req.file.path}`;

      await Face.deleteMany();

      await executePythonScript(videoPath);

      console.log('Script execution completed successfully');

      await postImagesToDatabase('scripts/faces');

      setTimeout(async () => {
        const faces = await Face.find();

        // Convert the image data to base64 before sending it to the frontend
        const formattedFaces = faces.map((face) => ({
          _id: face._id,
          name: face.name,
          image: face.image.toString('base64'),
        }));

        res.status(200).json({ message: 'Video uploaded and script executed successfully', faces: formattedFaces });
      }, 3000);
    });
  } catch (error) {
    console.error('Error uploading and processing video:', error);
    res.status(500).json({ error: 'Error uploading and processing video' });
  }
};

const getData = async (req, res) => {
  try {
    const faces = await Face.find();

    // Convert the image data to base64 before sending it to the frontend
    const formattedFaces = faces.map((face) => ({
      _id: face._id,
      name: face.name,
      image: face.image.toString('base64'),
    }));

    res.status(200).json({ message: 'faces Fetched', faces: formattedFaces });
  } catch (error) {
    console.error('Error retrieving data:', error);
    res.status(500).json({ error: 'Error retrieving data' });
  }
};

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

      // Read the image file
      fs.readFile(imagePath, async (err, data) => {
        if (err) {
          console.error(`Error reading image ${file}:`, err);
          return;
        }

        // Create a new image document with the image data
        const image = new Face({
          name: path.parse(file).name,
          image: data,
        });

        try {
          // Save the image document to the database
          await image.save();
          console.log(`Image ${file} uploaded and saved to the database`);
        } catch (error) {
          console.error(`Error saving image ${file} to the database:`, error);
        }
      });
    });
  });
};

const executePythonScript = (videoPath) => {
  console.log('Script start Running');

  const pythonExecutablePath = `${process.env.PYTHON_EXE_PATH}`;
  const pythonScriptPath = 'scripts/test.py';

  return new Promise((resolve, reject) => {
    const process = spawn(pythonExecutablePath, [pythonScriptPath, videoPath]);

    process.stdout.on('data', (data) => {
      console.log(`Python script output: ${data}`);
    });

    process.stderr.on('data', (data) => {
      console.error(`Python script error: ${data}`);
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
  getData,
};
