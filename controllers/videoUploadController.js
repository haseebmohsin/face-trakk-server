const asyncHandler = require('express-async-handler');
const multer = require('multer');
const { spawn } = require('child_process');
const Face = require('../models/clusterModel');
const Cluster = require('../models/clusterModel');

const fs = require('fs');
const util = require('util');
const path = require('path');
const readdir = util.promisify(fs.readdir);

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

      await storeClustersDataInDatabase('scripts/clusters');

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

const storeClustersDataInDatabase = async (folderPath) => {
  try {
    const clusterDirs = await readdir(folderPath);

    for (const clusterDir of clusterDirs) {
      const clusterPath = path.join(folderPath, clusterDir);

      try {
        // Read the image files in the cluster directory
        const files = await readdir(clusterPath);

        // Store the face images and their data in an array
        const faceImagesData = [];
        for (const file of files) {
          const imagePath = path.join(clusterPath, file);

          // Read the image file as a Buffer
          const imageData = await fs.promises.readFile(imagePath);

          // Add the face image data to the array
          faceImagesData.push({
            imageName: file,
            faceImage: imageData,
          });
        }

        // Create a new cluster document with the face images data
        const cluster = new Cluster({
          clusterName: clusterDir,
          faceImages: faceImagesData,
        });

        // Save the cluster document to the database
        await cluster.save();
        console.log(`${clusterDir} uploaded and saved to the database`);
      } catch (error) {
        console.error(`Error storing ${clusterDir} and face images:`, error);
      }
    }
  } catch (err) {
    console.error('Error reading directory:', err);
  }
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
