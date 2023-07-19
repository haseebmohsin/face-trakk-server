const asyncHandler = require('express-async-handler');
const multer = require('multer');
const { spawn } = require('child_process');
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
 * @desc    Upload Video for cluster recognition
 * @route   POST /api/videoUpload
 * @access  Private
 */
const videoUpload = asyncHandler(async (req, res) => {
  upload.single('video')(req, res, async (err) => {
    if (err) {
      console.error('Error uploading file:', err);
      res.status(500);
      throw new Error('Error uploading file');
    }

    const videoPath = `${req.file.path}`;

    await Cluster.deleteMany();

    const pythonScriptPath = 'scripts/test.py';
    await executePythonScript(pythonScriptPath, [videoPath]);
    console.log('Script execution completed successfully');

    await storeClustersDataInDatabase('scripts/clusters');

    setTimeout(async () => {
      const clustersData = await getClustersDataFromDatabase();

      res.status(200).json({ message: 'Script executed successfully', clusters: clustersData });
    }, 3000);
  });
});

/**
 * @desc    Update the XMLs
 * @route   POST /api/xmlUpdate
 * @access  Private
 */
const xmlUpdate = asyncHandler(async (req, res) => {
  const { cluster_id, ...data } = req.body;
  const { xml_path, old_names_array, new_label } = data;

  const pythonScriptPath = 'scripts/xml_update.py';
  await executePythonScript(pythonScriptPath, [xml_path, old_names_array, new_label]);
  console.log('Script execution completed successfully');

  // Update the Cluster to set isActive to false for the specified cluster_id
  await Cluster.findOneAndUpdate({ _id: cluster_id }, { isActive: false });

  res.status(200).json({ message: 'XMLs Updated Successfully!' });
});

/**
 * @desc    Get clusters data
 * @route   GET /api/clusters
 * @access  Private
 */
const getClustersData = asyncHandler(async (req, res) => {
  const clustersData = await getClustersDataFromDatabase();

  res.status(200).json({ message: 'clusters Fetched', clusters: clustersData });
});

/**
 * @desc    Get cluster data by ID
 * @route   GET /api/clusters/:id
 * @access  Private
 */
const getClusterDataById = asyncHandler(async (req, res) => {
  const clusterId = req.params.id;
  const cluster = await Cluster.findById(clusterId);

  if (!cluster) {
    res.status(404);
    throw new Error('Cluster not found');
  }

  // Convert the image data to base64 before sending it to the frontend
  const clusterData = {
    _id: cluster._id,
    clusterName: cluster.clusterName,
    faceImagesArray: cluster.faceImagesArray.map((item) => ({
      faceName: item.faceName,
      faceImage: item.faceImage.toString('base64'),
    })),
  };

  res.status(200).json({ cluster: clusterData });
});

/**
 * @desc    Get clusters data from the database
 * @returns {Promise<Array>} Array of cluster data
 */
const getClustersDataFromDatabase = async () => {
  const clusters = await Cluster.find({ isActive: true });

  // Convert the image data to base64 before sending it to the frontend
  const clustersData = clusters.map((cluster) => ({
    _id: cluster._id,
    clusterName: cluster.clusterName,
    faceImagesArray: cluster.faceImagesArray.map((item) => ({
      faceName: item.faceName,
      faceImage: item.faceImage.toString('base64'),
    })),
  }));

  return clustersData;
};

/**
 * @desc    Store clusters data in the database
 * @param   {string} folderPath - Path of the folder containing cluster data
 * @returns {Promise<void>}
 */
const storeClustersDataInDatabase = async (folderPath) => {
  const clusterDirs = await readdir(folderPath);

  for (const clusterDir of clusterDirs) {
    const clusterPath = path.join(folderPath, clusterDir);

    try {
      // Read the image files in the cluster directory
      const files = await readdir(clusterPath);

      // Store the cluster images and their data in an array
      const faceImagesData = [];
      for (const file of files) {
        const imagePath = path.join(clusterPath, file);

        // Read the image file as a Buffer
        const imageData = await fs.promises.readFile(imagePath);

        // Add the cluster image data to the array
        faceImagesData.push({
          faceName: path.parse(file).name,
          faceImage: imageData,
        });
      }

      // Create a new cluster document with the cluster images data
      const cluster = new Cluster({
        clusterName: clusterDir,
        faceImagesArray: faceImagesData,
      });

      // Save the cluster document to the database
      await cluster.save();
      console.log(`${clusterDir} uploaded and saved to the database`);
    } catch (error) {
      console.error(`Error storing ${clusterDir} and cluster images:`, error);
    }
  }
};

/**
 * @desc    Execute Python script
 * @param   {string} pythonScriptPath - Path of the Python script file
 * @param   {Object} args - Arguments to be passed to the `spawn` function
 * @returns {Promise<void>}
 */
const executePythonScript = (pythonScriptPath, args) => {
  console.log('Script start Running');

  const pythonExecutablePath = `${process.env.PYTHON_EXE_PATH}`;

  return new Promise((resolve, reject) => {
    const process = spawn(pythonExecutablePath, [pythonScriptPath, ...args]);

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
        console.log('Script execution failed for some reason');

        reject(new Error('Script execution failed'));
      }
    });
  });
};

module.exports = {
  videoUpload,
  getClustersData,
  getClusterDataById,
  xmlUpdate,
};
