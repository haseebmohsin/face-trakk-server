const asyncHandler = require('express-async-handler');
const multer = require('multer');
const { spawn } = require('child_process');
const Cluster = require('../models/clusterModel');
const { upload } = require('../utils/fileUtils');
const { executePythonScript } = require('../utils/pythonUtils');

const fs = require('fs');
const util = require('util');
const path = require('path');
const readdir = util.promisify(fs.readdir);

/**
 * @desc    Upload Video for cluster recognition
 * @route   POST /api/uploadVideo
 * @access  Private
 */
const uploadVideo = asyncHandler(async (req, res) => {
  upload.single('video')(req, res, async (err) => {
    if (err) {
      console.error('Error uploading file:', err);
      res.status(500);
      throw new Error('Error uploading file');
    }

    const videoPath = `${req.file.path}`;

    await Cluster.deleteMany();

    const pythonScriptPath = 'scripts/recognize.py';
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
 * @desc    Move faces to another cluster or create a new cluster
 * @route   POST /api/moveToNewCluster
 * @access  Private
 */
const moveToNewCluster = asyncHandler(async (req, res) => {
  const { clusterId, selectedItemIds } = req.body;

  let sourceCluster = await Cluster.findOne({ 'faceImagesArray._id': { $in: selectedItemIds } });
  let SelectedImages = sourceCluster.faceImagesArray.filter((item) => selectedItemIds.includes(item._id.toString()));

  if (clusterId) {
    const targetCluster = await Cluster.findById(clusterId);

    if (sourceCluster._id.toString() === targetCluster._id.toString()) {
      res.status(400);
      throw new Error('The Images are already in the same cluster');
    }

    targetCluster.faceImagesArray = [...targetCluster.faceImagesArray, ...SelectedImages];

    // Save changes to the target cluster
    await targetCluster.save();

    res.status(200).json({ message: 'Moved successfully' });
  } else {
    // Create a new cluster and add the selected face images to it
    const newCluster = new Cluster({
      faceImagesArray: SelectedImages,
    });

    // Save the new cluster
    await newCluster.save();
  }

  // Remove the selected face images from the source cluster
  sourceCluster.faceImagesArray = sourceCluster.faceImagesArray.filter((item) => !selectedItemIds.includes(item._id.toString()));

  if (sourceCluster.faceImagesArray.length === 0) {
    // If there are no more face images in the source cluster, remove the entire cluster
    await sourceCluster.remove();
  } else {
    // Save the changes to the source cluster
    await sourceCluster.save();
  }

  res.status(200).json({ message: 'Moved successfully' });
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
      _id: item._id,
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
      _id: item._id,
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
 * @desc    execute the training script
 * @route   POST /api/createAdditionalTrainingDatasets
 * @access  Private
 */
const createAdditionalTrainingDatasets = asyncHandler(async (req, res) => {
  const { cluster_id, selectedItemIds, label } = req.body;

  // Find the cluster based on cluster_id
  const cluster = await Cluster.findById(cluster_id);

  if (!cluster) {
    return res.status(404).json({ message: 'Cluster not found' });
  }

  // Create a directory with the provided label inside additional-training-datasets
  const directoryPath = path.join(__dirname, '..', 'scripts', 'database', 'additional-training-datasets', label);

  if (!fs.existsSync(directoryPath)) {
    fs.mkdirSync(directoryPath);
  }

  // Loop through the selectedItemIds and write the face images to the directory
  for (const itemId of selectedItemIds) {
    const selectedImage = cluster.faceImagesArray.find((image) => image._id.toString() === itemId);

    if (selectedImage) {
      const imageBuffer = selectedImage.faceImage;
      const imageFilePath = path.join(directoryPath, `${itemId}.jpg`);

      fs.writeFileSync(imageFilePath, imageBuffer);
    }
  }

  res.status(200).json({ message: 'Training Dataset is created' });
});

/**
 * @desc    execute the training script
 * @route   POST /api/startTraining
 * @access  Private
 */
const startTraining = asyncHandler(async (req, res) => {
  const pythonScriptPath = 'scripts/train.py';

  await executePythonScript(pythonScriptPath);
  console.log('Script execution completed successfully');

  res.status(200).json({ message: 'Training Completed!' });
});

module.exports = {
  uploadVideo,
  getClustersData,
  getClusterDataById,
  createAdditionalTrainingDatasets,
  startTraining,
  moveToNewCluster,
};
